# 生产部署

本方案面向 Linux x86_64 单机服务器，使用 Docker Compose 部署一个 API
进程。任务队列位于进程内，任务状态保存在本机卷，因此不能增加 Uvicorn worker
数量，也不能通过 `docker compose up --scale app=...` 横向扩容。应用启动时还会在任务
卷中获取排他文件锁，误启动第二个实例会直接失败。

## 1. 拓扑和前置条件

```text
可信内网客户端
      │ HTTP :8080
      ▼
Docker 端口映射
      │
      ▼
Uvicorn :8000（1 个进程，直接对外提供 API）
      ├── jobs 卷：上传文件、状态、事件和结果
      └── models 卷：PaddleOCR 模型缓存
```

服务器需要：

- Linux x86_64。
- Docker Engine 和 Docker Compose V2。
- 能访问配置的 LLM 服务。
- 首次预热时能访问 Paddle 模型源。
- GPU 部署额外需要兼容 CUDA 11.8 的 NVIDIA 驱动和 NVIDIA Container
  Toolkit；`nvidia-smi` 和容器 GPU 访问必须正常。

本部署不提供 TLS 或访问认证。端口必须只绑定服务器内网地址，并由主机防火墙阻止公网
或不可信网段访问。

## 2. 配置

创建本机配置文件：

```bash
cp deploy/env.production.example deploy/.env.production
chmod 600 deploy/.env.production
```

编辑 `deploy/.env.production`：

- 将 `EXAM_REC_BIND_IP` 改成服务器的可信内网 IP；默认 `127.0.0.1` 只允许本机访问。
- 设置实际的 `EXAM_REC_LLM_API_KEY`、`EXAM_REC_LLM_BASE_URL` 和
  `EXAM_REC_LLM_MODEL`。
- 根据磁盘和处理能力调整上传、页数、排队数及 CPU 线程数限制。
- `EXAM_REC_LOG_LEVEL` 默认是 `INFO`，可设为 `DEBUG`、`WARNING`、`ERROR`
  或 `CRITICAL`。

`deploy/.env.production` 已被 Git 忽略，必须保持 `0600` 权限。密钥会作为容器环境
变量传入，因此拥有 Docker 管理权限的用户可以通过容器检查命令读取；本方案假定 Docker
权限只授予可信管理员。

上传大小由应用的 `EXAM_REC_MAX_UPLOAD_BYTES` 限制。

## 3. 首次部署

确保所有需要发布的变更已经提交，工作区中的已跟踪文件没有改动，然后执行：

```bash
# 普通服务器
./deploy/release.sh cpu

# NVIDIA GPU 服务器
./deploy/release.sh gpu
```

脚本会：

1. 根据 `uv.lock` 构建 `exam-rec:<git-sha>-cpu` 或 `-gpu` 镜像。
2. 验证任务卷、模型卷、LLM 配置和 Paddle 运行时。
3. 在持久模型卷中完成一次最小 OCR 预热。
4. 启动一个 Uvicorn 进程并将配置的宿主机端口直接映射到容器。
5. 等待 `/health/ready` 就绪并从应用容器执行健康检查。

GPU 预检要求 Paddle 是 CUDA 构建且至少有一张可见 GPU；CPU 镜像如果误装了 CUDA
Paddle 也会拒绝启动，不会静默回退。

## 4. 日常检查

选择与部署时相同的 Compose 文件。以下以 CPU 为例：

```bash
docker compose --env-file deploy/.env.production ps
docker compose --env-file deploy/.env.production logs --tail 200 app
curl http://SERVER_LAN_IP:8080/health/live
curl http://SERVER_LAN_IP:8080/health/ready
```

GPU 部署的直接 Compose 命令需要额外添加：

```text
-f compose.yaml -f compose.gpu.yaml
```

健康接口含义：

- `/health/live` 返回 200 表示 Web 进程存活。
- `/health/ready` 只在识别 worker 可用时返回 200，否则返回 503。
- `/health` 保留原有兼容行为，无论 worker 是否可用都返回 200，客户端需要检查
  JSON 状态。

应用日志以单行文本写到 stdout，包含识别服务和任务的关键生命周期；Uvicorn 请求日志
也位于 `app` 容器。持续查看：

```bash
docker compose --env-file deploy/.env.production logs -f app
```

容器日志采用 Docker `json-file` 驱动，每个文件最大 50 MB，保留 5 个文件。任务的
`events.jsonl` 仍是持久化状态记录，不会被应用日志替代。

## 5. 停止和恢复服务

停止并删除 CPU 部署的容器和 Compose 网络：

```bash
docker compose \
  --env-file deploy/.env.production \
  -f compose.yaml \
  down
```

GPU 部署需要同时指定 GPU 覆盖文件：

```bash
docker compose \
  --env-file deploy/.env.production \
  -f compose.yaml \
  -f compose.gpu.yaml \
  down
```

`down` 不会删除 `exam-rec_jobs` 和 `exam-rec_models` 命名卷。不要添加 `-v`，否则
任务数据和已下载模型会被一并删除。重新部署时执行 `./deploy/release.sh cpu` 或
`./deploy/release.sh gpu`。

如果只是临时停机并希望保留现有容器，使用 `stop`，之后使用相同 Compose 参数执行
`start`。以下以 CPU 部署为例：

```bash
docker compose --env-file deploy/.env.production -f compose.yaml stop
docker compose --env-file deploy/.env.production -f compose.yaml start
```

停止服务前应确认没有活跃识别任务。手动停机不会自动恢复未完成任务；如果需要升级版本，
应使用发布脚本提供的排空和回滚流程。

## 6. 更新和回滚

更新服务器源码到目标提交后，重新运行对应的 `release.sh`。脚本首先构建新镜像，然后：

1. 最多等待一小时，直到所有 `queued`、`running` 和 `cancelling` 任务结束。
2. 优雅停止旧应用，预热新镜像并启动新版本。
3. 等待新应用通过容器内健康检查。

应用端口直接对外提供服务，发布脚本等待任务排空期间仍可能收到新任务。因此升级前应先
暂停调用方提交任务。部署期间会有短暂不可用时间。不要在任务活跃时手动执行
`docker compose down`；强制重启会使未完成任务变成 `interrupted`，且不会自动恢复。

新版本未能健康启动时，发布脚本会自动恢复升级前的镜像。需要主动回滚时，将源码切换到
目标提交并重新执行 `release.sh`；任务卷和模型卷不会被替换。

## 7. 数据和容量

Docker 命名卷：

- `exam-rec_jobs`：`input.pdf`、`status.json`、`events.jsonl` 和
  `result.json`。
- `exam-rec_models`：PaddleX/PaddleOCR 下载的模型。

任务数据不会自动过期。客户端不再需要结果时，应调用
`DELETE /recognitions/{job_id}`。运维侧需要监控 Docker 数据盘：

- 使用率达到 80% 时告警并安排清理。
- 使用率达到 90% 时停止接收新任务并立即处理。

检查容量：

```bash
docker system df -v
docker volume inspect exam-rec_jobs exam-rec_models
df -h /var/lib/docker
```

不要通过宿主机直接删除正在处理的任务目录。当前方案不包含自动清理、自动备份、集中日志、
Prometheus 或高可用调度。
