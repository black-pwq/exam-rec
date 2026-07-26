# 题本识别 Web API

本服务接收 PDF 题本，将任务放入 FIFO 队列，由单个后台 worker 依次识别。
客户端上传成功后不需要保持连接，可以使用任务 ID 查询状态，并通过游标分批获取新增的
页面识别结果。

## 1. 服务地址与约定

本文示例假设服务地址为：

```text
http://localhost:8000
```

通用约定：

- 除文件上传外，请求和响应均使用 JSON。
- 所有 PDF 页索引都是零基索引。
- 时间字段是带时区的 ISO 8601 字符串。
- `job_id` 是服务生成的 32 位十六进制字符串。
- 当前版本没有身份认证，建议只部署在可信内网。
- 当前版本不提供流式响应和任务断点恢复。

## 2. 任务处理流程

推荐客户端按以下顺序调用：

1. 调用 `POST /recognitions` 上传 PDF，保存返回的 `job_id`。
2. 定时调用 `GET /recognitions/{job_id}` 显示任务状态和总体进度。
3. 定时调用 `GET /recognitions/{job_id}/updates` 获取新增页面，保存
   `next_cursor`。
4. 任务变成 `completed` 后，调用 `GET /recognitions/{job_id}/result`
   获取完整题目列表。
5. 不再需要任务数据时，调用 `DELETE /recognitions/{job_id}`。

任务按照先来先到顺序执行。同一时间只处理一份题本，后续任务保持 `queued`。

## 3. 健康检查

### `GET /health`

检查后台识别线程是否正常运行。

成功响应：

```http
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{
  "status": "ok"
}
```

如果后台 worker 不可用，返回体中的状态为：

```json
{
  "status": "unavailable"
}
```

注意：当前健康检查无论 worker 是否可用都返回 HTTP 200，客户端或部署平台需要检查
JSON 中的 `status`。

## 4. 上传并创建任务

### `POST /recognitions`

使用 `multipart/form-data` 上传 PDF。文件字段名必须是 `file`。

请求示例：

```bash
curl -X POST http://localhost:8000/recognitions \
  -F 'file=@questions.pdf;type=application/pdf'
```

成功响应：

```http
HTTP/1.1 202 Accepted
Content-Type: application/json
Location: /recognitions/6c84eb1bc7044ea1b9555b06bb47c22f
```

```json
{
  "job_id": "6c84eb1bc7044ea1b9555b06bb47c22f",
  "status": "queued",
  "file_size": 58320124,
  "page_count": 300
}
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `job_id` | string | 后续所有请求使用的任务 ID |
| `status` | string | 新任务固定为 `queued` |
| `file_size` | integer | 上传文件的字节数 |
| `page_count` | integer | PDF 总页数 |

响应头 `Location` 是任务状态查询地址。客户端可以保存响应体中的 `job_id`，也可以
直接使用 `Location` 查询任务状态；增量结果和完整结果接口仍按下文约定由 `job_id`
拼接。

上传限制：

- 默认最大文件大小为 500 MB。
- 默认最大页数为 500 页。
- 不接受空文件、非 PDF、空 PDF 或密码保护 PDF。
- 默认最多允许 32 个任务等待。

可能的错误：

| HTTP 状态码 | 场景 |
| --- | --- |
| `413` | 文件大小或 PDF 页数超过限制 |
| `422` | 缺少 `file` 字段、文件为空、PDF 无效或 PDF 受密码保护 |
| `429` | FIFO 等待队列已满，响应带有 `Retry-After: 30` |
| `503` | 后台识别 worker 不可用 |

前端 JavaScript 示例：

```javascript
async function createRecognition(file) {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch("/recognitions", {
    method: "POST",
    body: form,
  });

  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail ?? "题本上传失败");
  }
  return body;
}
```

## 5. 查询任务状态

### `GET /recognitions/{job_id}`

请求示例：

```bash
curl http://localhost:8000/recognitions/6c84eb1bc7044ea1b9555b06bb47c22f
```

响应示例：

```json
{
  "job_id": "6c84eb1bc7044ea1b9555b06bb47c22f",
  "status": "running",
  "original_filename": "questions.pdf",
  "page_count": 300,
  "processed_pages": 38,
  "problem_count": 152,
  "error": null,
  "created_at": "2026-07-25T02:10:15.142000+00:00",
  "updated_at": "2026-07-25T02:11:03.571000+00:00"
}
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `status` | string | 当前任务状态 |
| `original_filename` | string | 上传时的原始文件名，不包含客户端路径 |
| `page_count` | integer | 原 PDF 总页数 |
| `processed_pages` | integer | 已产生页面结果的题目区间页数 |
| `problem_count` | integer | 当前累计得到的完整题目数 |
| `error` | string/null | 失败或中断原因 |
| `created_at` | string | 创建时间 |
| `updated_at` | string | 最后状态更新时间 |

`processed_pages` 不是当前 PDF 页索引，也不一定等于 `page_index + 1`。题目识别可能从
封面和目录之后开始，并且跨页题目只有确认完整后才产生结果。

任务状态：

| 状态 | 是否终态 | 说明 |
| --- | --- | --- |
| `queued` | 否 | 正在等待前面的任务完成 |
| `running` | 否 | 正在识别 |
| `cancelling` | 否 | 已请求删除，等待当前处理步骤退出 |
| `completed` | 是 | 识别完成，可获取最终结果 |
| `failed` | 是 | OCR、LLM 或结构化提取失败 |
| `cancelled` | 是 | 任务已取消 |
| `interrupted` | 是 | 服务在任务完成前停止，任务不会自动恢复 |

不存在的任务返回：

```http
HTTP/1.1 404 Not Found
```

```json
{
  "detail": "job not found"
}
```

## 6. 获取增量页面结果

### `GET /recognitions/{job_id}/updates`

查询参数：

| 参数 | 类型 | 默认值 | 限制 | 说明 |
| --- | --- | --- | --- | --- |
| `after` | integer | `0` | 大于等于 0 | 只返回此序号之后的事件 |
| `limit` | integer | `100` | 1 到 500 | 单次最多返回的事件数 |

首次请求：

```bash
curl 'http://localhost:8000/recognitions/6c84eb1bc7044ea1b9555b06bb47c22f/updates?after=0&limit=100'
```

响应：

```json
{
  "job_id": "6c84eb1bc7044ea1b9555b06bb47c22f",
  "status": "running",
  "next_cursor": 15,
  "has_more": false,
  "events": [
    {
      "sequence": 14,
      "type": "page",
      "created_at": "2026-07-25T02:11:02.120000+00:00",
      "page_index": 26,
      "problems": []
    },
    {
      "sequence": 15,
      "type": "page",
      "created_at": "2026-07-25T02:11:03.570000+00:00",
      "page_index": 27,
      "problems": []
    }
  ]
}
```

客户端必须保存 `next_cursor`，下一次请求使用：

```text
GET /recognitions/{job_id}/updates?after=15
```

响应字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `status` | string | 生成本次响应时的任务状态 |
| `next_cursor` | integer | 下一次请求应传入的 `after` |
| `has_more` | boolean | 本次是否因 `limit` 截断，仍有历史事件可立即读取 |
| `events` | array | 按 `sequence` 升序排列的事件 |

轮询规则：

1. 初始游标使用 `0`。
2. 每次请求成功后立即保存 `next_cursor`。
3. 如果 `has_more=true`，立即再次请求，不需要等待。
4. 如果 `has_more=false` 且任务仍在运行，建议等待 1 到 3 秒后再次请求。
5. 即使响应中的任务状态已经是终态，也应先读取到 `has_more=false`，避免漏掉终态前的
   页面事件。

### 事件公共字段

所有事件都包含：

```json
{
  "sequence": 15,
  "type": "page",
  "created_at": "2026-07-25T02:11:03.570000+00:00"
}
```

- `sequence` 在单个任务内从 1 开始严格递增。
- 游标应使用 `sequence`，不能使用 `page_index`。
- `created_at` 是事件写入本地文件的时间。

### `queued` 事件

任务已持久化并放入等待队列：

```json
{
  "sequence": 1,
  "type": "queued",
  "created_at": "2026-07-25T02:10:15.142000+00:00"
}
```

### `started` 事件

后台 worker 开始处理该任务：

```json
{
  "sequence": 2,
  "type": "started",
  "created_at": "2026-07-25T02:10:20.100000+00:00"
}
```

### `page` 事件

产生一页结构化结果：

```json
{
  "sequence": 15,
  "type": "page",
  "created_at": "2026-07-25T02:11:03.570000+00:00",
  "page_index": 27,
  "problems": [
    {
      "number": "15",
      "question": "下列说法正确的是……",
      "answer": "",
      "options": {
        "A": "选项一",
        "B": "选项二",
        "C": "选项三",
        "D": "选项四"
      },
      "analysis": ""
    }
  ],
  "extractor_name": "GeneralRegexExtractor",
  "evaluation": {
    "score": 0.94,
    "metrics": {
      "question_completeness": 1.0,
      "option_completeness": 1.0
    },
    "warnings": []
  }
}
```

`problems` 可能为空。当前提取器允许题目跨页，因此某页 OCR 成功后不一定立即确认一
道完整题目。前端应按事件顺序把非空的 `problems` 追加到展示列表，不要假设每页至少有
一道题。

题目字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `number` | string | 题号，可能包含非纯数字格式 |
| `question` | string | 题干 |
| `answer` | string | 参考答案；题本没有答案时为空字符串 |
| `options` | object | 选项标签到选项文本的映射 |
| `analysis` | string | 解析；题本没有解析时为空字符串 |

### `completed` 事件

任务成功完成：

```json
{
  "sequence": 305,
  "type": "completed",
  "created_at": "2026-07-25T02:16:30.000000+00:00",
  "problem_count": 1200
}
```

### `error` 事件

任务执行失败：

```json
{
  "sequence": 23,
  "type": "error",
  "created_at": "2026-07-25T02:11:20.000000+00:00",
  "error_type": "LowConfidenceQuestionRangeError",
  "message": "question start confidence 0.52 is below the minimum 0.70"
}
```

当前版本发生执行错误后会停止整个任务。此前已经写入的页面事件仍可通过游标读取，但
不会生成最终 `result.json`。

### 取消和中断事件

正在运行的任务被删除时会先产生：

```json
{
  "sequence": 18,
  "type": "cancellation_requested",
  "created_at": "2026-07-25T02:11:10.000000+00:00"
}
```

任务退出时可能产生 `cancelled`。服务启动时发现上次未完成的任务，会产生
`interrupted`。删除流程最终会移除整个任务目录，因此客户端随后查询可能直接得到
404。

### 前端轮询示例

```javascript
const TERMINAL_STATUSES = new Set([
  "completed",
  "failed",
  "cancelled",
  "interrupted",
]);

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function pollRecognition(jobId, onEvent) {
  let cursor = 0;

  while (true) {
    const url = new URL(
      `/recognitions/${encodeURIComponent(jobId)}/updates`,
      window.location.origin,
    );
    url.searchParams.set("after", String(cursor));
    url.searchParams.set("limit", "100");

    const response = await fetch(url);
    if (!response.ok) {
      const body = await response.json();
      throw new Error(body.detail ?? "获取识别进度失败");
    }

    const update = await response.json();
    for (const event of update.events) {
      onEvent(event);
    }
    cursor = update.next_cursor;

    if (update.has_more) {
      continue;
    }
    if (TERMINAL_STATUSES.has(update.status)) {
      return update.status;
    }
    await delay(2000);
  }
}
```

浏览器刷新后，可以将 `job_id` 和最近的 `cursor` 保存在应用状态或
`localStorage` 中，再从该游标继续请求。如果游标没有保存，从 `after=0` 重新读取也
不会影响服务端任务，只需要前端避免重复追加相同 `sequence` 的事件。

## 7. 获取完整结果

### `GET /recognitions/{job_id}/result`

只允许对 `completed` 任务调用。

成功响应：

```json
{
  "job_id": "6c84eb1bc7044ea1b9555b06bb47c22f",
  "status": "completed",
  "problem_count": 1200,
  "problems": [
    {
      "number": "1",
      "question": "题干……",
      "answer": "",
      "options": {
        "A": "选项一",
        "B": "选项二"
      },
      "analysis": ""
    }
  ]
}
```

任务尚未完成时返回：

```http
HTTP/1.1 409 Conflict
```

```json
{
  "detail": {
    "message": "recognition job is not complete: running",
    "status": "running"
  }
}
```

不存在的任务返回 404。

## 8. 取消并删除任务

### `DELETE /recognitions/{job_id}`

删除排队中、已完成、失败或中断的任务时，本地目录会立即删除：

```http
HTTP/1.1 202 Accepted
```

```json
{
  "job_id": "6c84eb1bc7044ea1b9555b06bb47c22f",
  "status": "deleted"
}
```

删除正在运行的任务时，先请求取消：

```json
{
  "job_id": "6c84eb1bc7044ea1b9555b06bb47c22f",
  "status": "cancelling"
}
```

OCR 或提取器不能在任意指令处强制中断，任务会在当前可中断步骤结束后退出并删除本地
目录。之后状态、更新和结果接口都返回 404。

删除接口不是幂等的：对已经删除的任务再次调用会返回 404。

## 9. 本地文件

每个任务对应一个目录：

```text
var/jobs/<job_id>/
├── input.pdf
├── status.json
├── events.jsonl
└── result.json
```

- `input.pdf`：上传文件。
- `status.json`：当前状态和进度。
- `events.jsonl`：按序号追加的增量事件。
- `result.json`：成功完成后生成的完整题目列表。

文件默认一直保留，直到调用删除接口。部署方需要监控磁盘使用量。

如果服务重启，之前处于 `queued`、`running` 或 `cancelling` 的任务会被标记为
`interrupted`，但不会自动重新开始。已经完成的任务和已有增量事件仍然可以查询。

## 10. 服务配置与启动

识别 worker 必须配置：

```text
EXAM_REC_LLM_API_KEY
EXAM_REC_LLM_BASE_URL
EXAM_REC_LLM_MODEL
```

可选配置：

```text
EXAM_REC_JOB_ROOT=var/jobs
EXAM_REC_MAX_UPLOAD_BYTES=524288000
EXAM_REC_MAX_PDF_PAGES=500
EXAM_REC_MAX_QUEUED_JOBS=32
```

使用一个 Web 进程启动：

```bash
uv run uvicorn api:app --workers 1
```

接口文档还可以通过 FastAPI 自动生成的页面查看：

```text
http://localhost:8000/docs
http://localhost:8000/redoc
```
