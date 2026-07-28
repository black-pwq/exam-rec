# 生产部署

本项目使用 Docker Compose 在 Linux x86_64 单机上运行一个 Uvicorn
进程。任务状态和模型缓存在 Docker 命名卷中；应用使用进程内任务队列，因此不要增加
Uvicorn worker 数量或横向扩容 `app` 服务。

## 1. 配置

在项目根目录创建环境文件：

```bash
cp .env.example .env
chmod 600 .env
```

至少配置以下 LLM 参数：

```text
EXAM_REC_LLM_API_KEY
EXAM_REC_LLM_BASE_URL
EXAM_REC_LLM_MODEL
```

`.env` 同时用于 Compose 变量替换和容器环境注入。应用也会在当前工作目录读取
`.env`，但不会覆盖操作系统或 Compose 已提供的环境变量。

`EXAM_REC_BIND_IP` 默认是 `127.0.0.1`。只有在主机防火墙和访问控制已经配置妥当时，
才应绑定其他地址；服务本身不提供 TLS 或访问认证。

## 2. 启动

CPU 部署：

```bash
docker compose config --quiet
docker compose up --detach --build
```

NVIDIA CUDA 11.8 部署：

```bash
docker compose \
  -f compose.yaml \
  -f compose.gpu.yaml \
  config --quiet
docker compose \
  -f compose.yaml \
  -f compose.gpu.yaml \
  up --detach --build
```

GPU 主机需要兼容驱动、NVIDIA Container Toolkit，并确保容器能够访问 GPU。

## 3. 检查与日志

```bash
docker compose ps
docker compose logs --tail 200 app
curl http://127.0.0.1:8080/health/live
curl http://127.0.0.1:8080/health/ready
```

镜像内健康检查请求 `/health/ready`。应用日志和 Uvicorn 请求日志由 Docker
`json-file` 驱动保存，每个文件最大 50 MB，保留 5 个文件。

健康接口含义：

- `/health/live` 返回 200 表示 Web 进程存活。
- `/health/ready` 只在识别 worker 可用时返回 200。
- `/health` 是兼容接口；客户端还需检查响应中的 `status`。

## 4. 更新与停止

当前部署不执行自动任务排空、模型预热或回滚。更新前应暂停新任务并确认没有
`queued`、`running` 或 `cancelling` 任务，然后重新运行对应的 `docker compose
up --detach --build` 命令。

停止并删除容器和网络：

```bash
docker compose down
```

GPU 部署执行 `down` 时同样添加 `compose.gpu.yaml`。不要使用 `down -v`，否则会删除
任务和模型卷。普通 `down` 不会删除：

- `exam-rec_jobs`：上传文件、任务状态、事件和结果。
- `exam-rec_models`：PaddleOCR 模型缓存。

如果服务在任务执行中被重建，未完成任务会在下次启动时标记为 `interrupted`，不会自动
恢复。

## 5. 数据容量

任务数据不会自动过期。客户端不再需要结果时，应调用
`DELETE /recognitions/{job_id}`。运维侧应监控 Docker 数据盘：

```bash
docker system df -v
docker volume inspect exam-rec_jobs exam-rec_models
df -h /var/lib/docker
```

不要直接删除正在处理的任务目录。当前简单部署不包含自动清理、备份、集中日志、
Prometheus 或高可用调度。
