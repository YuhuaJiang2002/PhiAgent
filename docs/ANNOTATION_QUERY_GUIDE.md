# 数据标注查询与无效样本构建指南

这份文档说明如何把数据库中的数据集审核结果，与本地 LeRobot 文件合并，生成一个包含“审核无效”样本的清单。示例使用项目 7、数据集 9677；将项目 ID、数据库名和路径替换为实际值即可复用。

## 1. 先区分三个字段

构建数据集时不要把下面三个概念混在一起：

| 字段 | 含义 | 典型来源 |
| --- | --- | --- |
| `task_label` | 任务内容，例如“打开医药箱……”或“bad盒子重复打开” | `meta/episodes.jsonl`、`meta/tasks.jsonl` |
| `annotation_status` | 是否已经完成标注，例如“已标注” | 标注系统数据库 |
| `review_status` | 审核是否有效，例如“有效”或“无效” | 标注系统数据库 |

`bad` 出现在任务文本中时，只表示任务/内容标签，不等于审核状态为“无效”。筛选无效数据应使用 `review_status`，再把 `task_label` 作为单独字段保留。

界面状态写入 JSON 时使用固定的英文值，避免后续筛选时混用中文和布尔值：

| 界面显示 | JSON 字段和值 |
| --- | --- |
| 已标注 | `"annotation_status": "annotated"` |
| 有效 | `"review_status": "valid"` |
| 无效 | `"review_status": "invalid"` |

因此，截图中的 Episode 4 应写为：

```json
{
  "episode_index": 4,
  "task_label": "bad盒子重复打开",
  "annotation_status": "annotated",
  "review_status": "invalid"
}
```

截图中的 Episode 3 则应写为：

```json
{
  "episode_index": 3,
  "task_label": "打开医药箱,取出并检查每种物品的保质期,然后放入,再检查下一个,最后扣上盖子",
  "annotation_status": "annotated",
  "review_status": "valid"
}
```

`annotation_status` 和 `review_status` 是两个独立字段：已标注但审核无效的数据仍然保留 `annotated`，只把 `review_status` 设为 `invalid`。

## 2. 从 MySQL 查询数据集和存储路径

查询项目 7 下的全部数据集：

```sql
SELECT
    d.id,
    d.name,
    d.storage_path,
    d.collection_task_id
FROM datasets AS d
JOIN collection_tasks AS ct ON ct.id = d.collection_task_id
WHERE ct.project_id = 7
ORDER BY d.id;
```

查询单个数据集 9677：

```sql
SELECT id, name, storage_path
FROM datasets
WHERE id = 9677;
```

当前示例的结果是：

```text
id            9677
name          20260831T1886-C260
storage_path  lerobot/CUS000001/20260831T1886-C260
```

本地真实目录由下面的根目录和 `storage_path` 拼接得到：

```text
/mnt/ldp_uploads/ldp-uploads/datasets/<storage_path>
```

因此 9677 的目录为：

```text
/mnt/ldp_uploads/ldp-uploads/datasets/lerobot/CUS000001/20260831T1886-C260
```

## 3. 找到审核状态所在的数据库表

`datasets` 表通常只保存数据集基本信息。审核状态的表名可能因部署版本不同而变化。先用字段名搜索候选表：

```sql
SELECT
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema = DATABASE()
  AND (
      column_name REGEXP '(^|_)(annotation|annotate|label|review|audit|valid|status)(_|$)'
  )
ORDER BY table_name, ordinal_position;
```

再查看候选表的结构和少量样例：

```sql
SHOW CREATE TABLE <candidate_table>;
SELECT * FROM <candidate_table> LIMIT 5;
```

重点寻找能关联到以下任意字段的表：

- `dataset_id` 或 `data_id`；
- `episode_index`、`episode_id` 或 `episode`；
- `annotation_status`、`label_status`；
- `review_status`、`audit_status`、`is_valid`；
- `invalid_reason`、`reject_reason`、`audit_remark`。

找到真实表名和字段后，按下面的逻辑查询，不要猜测表名：

```sql
SELECT
    d.id AS dataset_id,
    d.name AS dataset_name,
    d.storage_path,
    a.episode_index,
    a.annotation_status,
    a.review_status,
    a.invalid_reason,
    a.updated_at
FROM datasets AS d
JOIN <annotation_or_review_table> AS a
  ON a.dataset_id = d.id
WHERE d.id = 9677
ORDER BY a.episode_index;
```

如果状态是布尔字段，使用实际取值转换：

```sql
CASE
    WHEN a.is_valid = 1 THEN 'valid'
    WHEN a.is_valid = 0 THEN 'invalid'
    ELSE 'unknown'
END AS review_status
```

## 4. 从本地 LeRobot 文件读取 episode 标注

每个 `storage_path` 目录通常包含：

```text
<dataset> /
├── data/
│   └── chunk-*/file-*.parquet
├── meta/
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   ├── info.json
│   └── tasks.jsonl
└── videos/
    └── <video_key>/chunk-*/file-*.mp4
```

查询 Episode 4 的任务文本：

```bash
DATASET=/mnt/ldp_uploads/ldp-uploads/datasets/lerobot/CUS000001/20260831T1886-C260

sed -n '5p' "$DATASET/meta/episodes.jsonl"
cat "$DATASET/meta/tasks.jsonl"
```

也可以用标准库 Python 精确按 episode 查询：

```bash
python - "$DATASET/meta/episodes.jsonl" <<'PY'
import json
import sys

target = 4
with open(sys.argv[1], encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        if row.get("episode_index") == target:
            print(json.dumps(row, ensure_ascii=False, indent=2))
            break
PY
```

9677 的 Episode 4 当前本地内容为：

```json
{
  "episode_index": 4,
  "tasks": ["bad盒子重复打开"],
  "length": 5831
}
```

对应视频和轨迹文件：

```text
videos/observation.images.left_rgb/chunk-000/file-004.mp4
data/chunk-000/file-004.parquet
depths/observation.depth.left/chunk-000/episode-000004.mp4
```

实际使用前请用 `find` 检查文件是否存在，因为不同导出版本的视频命名可能不同：

```bash
find "$DATASET" -type f \( \
  -path '*file-004.mp4' -o \
  -path '*episode-000004.mp4' -o \
  -path '*file-004.parquet' \
\) -print
```

## 5. 生成“无效审核”样本清单

建议输出一行一个 episode 的 JSONL，而不是复制或移动原始视频。推荐字段如下：

```json
{
  "dataset_id": 9677,
  "dataset_name": "20260831T1886-C260",
  "storage_path": "lerobot/CUS000001/20260831T1886-C260",
  "episode_index": 4,
  "task_label": "bad盒子重复打开",
  "annotation_status": "annotated",
  "review_status": "invalid",
  "invalid_reason": "<来自审核系统的原因>",
  "video_path": "videos/observation.images.left_rgb/chunk-000/file-004.mp4",
  "depth_video_path": "depths/observation.depth.left/chunk-000/episode-000004.mp4",
  "parquet_path": "data/chunk-000/file-004.parquet"
}
```

多个 episode 放入 JSONL 时，每行是一个完整 JSON 对象，不能把多个对象包在一个数组里：

```jsonl
{"dataset_id":9677,"dataset_name":"20260831T1886-C260","episode_index":3,"task_label":"打开医药箱,取出并检查每种物品的保质期,然后放入,再检查下一个,最后扣上盖子","annotation_status":"annotated","review_status":"valid"}
{"dataset_id":9677,"dataset_name":"20260831T1886-C260","episode_index":4,"task_label":"bad盒子重复打开","annotation_status":"annotated","review_status":"invalid","invalid_reason":"审核页面显示无效"}
```

构建无效数据集时只筛选：

```python
row["annotation_status"] == "annotated" and row["review_status"] == "invalid"
```

建议保留 `invalid_reason`。如果审核系统没有提供原因，就写 `null` 或 `"审核页面显示无效"`，不要编造具体原因。

筛选规则建议：

```text
review_status in {"invalid", "无效", 0, false}
AND video_path 存在
AND parquet_path 存在
```

不要仅用 `task_label` 包含 `bad` 来筛选，因为这样会把“任务名称含 bad、但审核有效”的样本也混入无效集合。

如果审核表是一行一个 episode，可以直接在 SQL 中筛选：

```sql
SELECT
    d.id AS dataset_id,
    d.name AS dataset_name,
    d.storage_path,
    a.episode_index,
    a.annotation_status,
    a.review_status,
    a.invalid_reason
FROM datasets AS d
JOIN <annotation_or_review_table> AS a
  ON a.dataset_id = d.id
WHERE d.id IN (<dataset_ids>)
  AND (
      a.review_status IN ('invalid', '无效')
      OR a.is_valid = 0
  )
ORDER BY d.id, a.episode_index;
```

把这份查询结果导出为 TSV/CSV，再用 `storage_path` 拼接本地路径。生成清单时保留数据库原始状态值和查询时间，方便追溯。

## 6. 构建前的完整性检查

对每个候选 episode 至少检查：

```bash
test -f "$DATASET/meta/episodes.jsonl"
test -f "$DATASET/meta/tasks.jsonl"
test -f "$DATASET/data/chunk-000/file-004.parquet"
find "$DATASET/videos" -type f -name '*.mp4' | head
```

视频可以用 FFmpeg 验证：

```bash
ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames,duration \
  -of json \
  "$VIDEO"
```

还应检查三项数量是否一致：

1. `episodes.jsonl` 中的 `episode_index`；
2. 视频文件中的 episode/file 编号；
3. `data/` 中对应的 Parquet 文件。

对于 9677，Episode 4 的源文件映射为：

```text
source_mcap: 20260831_172742_ego.mcap
status:      success
parquet:     data/chunk-000/file-004.parquet
```

## 7. 已知的数据一致性问题

9677 的 Episode 4 中，`episodes.jsonl` 的任务文本是 `bad盒子重复打开`，但 `episodes_stats.jsonl` 中的 `task_index` 统计为 0。构建无效集合时应优先保留：

- 数据库中的审核状态；
- `episodes.jsonl` 中的 episode 任务文本；
- 原始 `task_index`，作为待修复字段保留。

不要在导出时静默改写原始 Parquet。可以在清单中额外记录：

```json
"task_index_consistent": false,
"task_index_observed": 0,
"task_label_source": "meta/episodes.jsonl"
```

## 8. 推荐输出目录

每次构建都使用独立目录，保留查询和环境信息：

```text
invalid-ego-dataset-YYYYMMDD-HHMMSS/
├── query_datasets.sql
├── query_reviews.sql
├── datasets.tsv
├── invalid_episodes.jsonl
├── validation_report.json
└── README.md
```

`validation_report.json` 至少记录：生成时间、数据库查询条件、数据集数量、episode 数量、有效视频数量、缺失文件数量和状态分布。原始视频和 Parquet 可以继续留在原始数据目录，通过清单中的相对路径引用。


## 9. 时间段标注：为什么构建清单里看不到 start/end

审核页面显示的时间段并没有消失，而是来自另一套字段：平台接口返回的
`annotationSegments`（数据库中通常对应 `annotation_segments_json`），每段包含
`startSec`、`endSec`、任务文本和保留/丢弃状态；有些版本还会返回
`splitMarkersSec`。`manifest.jsonl` 和 `meta/episodes.jsonl` 不是这个字段的导出物：

- `manifest.jsonl` 只记录审核状态、无效原因、文件路径等构建信息；
- `meta/episodes.jsonl` 的 `tasks` 只是该 episode 出现过的任务名称列表，不包含起止时间；
- 构建副本中的 `data/**/*.parquet` 仍保留逐帧 `task_index`，因此可以恢复帧级近似区间。

恢复本地区间时，必须用 `task_index` 读取 `meta/tasks.jsonl` 的文本映射，不要直接使用
Parquet 行里的 `task` 文本；部分导出版本的行级 `task` 字段是旧值或不准确的。把连续相同
`task_index` 的帧分组即可得到区间：

```python
import json
import pyarrow.parquet as pq

tasks = {}
with open(DATASET / "meta/tasks.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        tasks[int(row["task_index"])] = row["task"]

rows = pq.read_table(
    DATASET / "data/chunk-000/file-000.parquet",
    columns=["frame_index", "timestamp", "task_index"],
).to_pylist()

# 连续 task_index 的 [start, end) 分组；最后一段的结束时间用 episode_frames / fps。
```

这个恢复结果是按视频帧量化的近似值；若要与审核页面完全一致，应从审核接口导出
`annotationSegments`，不要用 `meta/episodes.jsonl` 反推。例如截图对应的
`dataset_id=8876, episode_index=0`：

```text
平台审核区间：任务 0:00.000–5:37.294；bad 5:37.294–5:39.800
本地逐帧恢复：任务 0:00.000–5:37.280；bad 5:37.280–5:39.800
```

二者只相差帧边界/时间戳取整，说明构建阶段没有丢掉逐帧信息；此前结果文件看不到区间，
是因为导出程序没有把 `annotationSegments` 或恢复后的 segments 字段写入清单。建议在
后续构建的每行增加：

```json
{
  "annotation_segments": [
    {"start_sec": 0.0, "end_sec": 337.294, "task_label": "...", "status": "keep"},
    {"start_sec": 337.294, "end_sec": 339.8, "task_label": "bad", "status": "drop"}
  ],
  "annotation_segments_source": "platform_api",
  "annotation_segments_exact": true
}
```

如果只能访问复制后的 LeRobot 文件，则将来源标为 `parquet_task_index`，并将
`annotation_segments_exact` 设为 `false`。
