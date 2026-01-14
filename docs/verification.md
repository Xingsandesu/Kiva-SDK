# Output Verification - 输出验证与自愈机制

Kiva SDK 提供了双层验证系统，自动校验 Agent 输出的质量，并在验证失败时触发自动重试。

## 为什么需要输出验证？

LLM Agent 的输出可能存在以下问题：

- 未完整回答用户问题
- 输出格式不符合预期
- 逻辑不完整或缺少关键信息
- 与分配的任务不匹配

输出验证系统作为"守门员"，确保只有高质量的输出才能返回给用户。

## 双层验证架构

Kiva 实现了两层验证：

### 1. Worker Agent 输出验证

在每个 Worker Agent 执行完成后立即验证：

- **验证目标**：Worker 被分配的具体任务（`task_assignment.task`）
- **验证内容**：
  - 输出是否响应了分配的具体任务
  - 输出是否包含足够的论据/逻辑支撑
  - 输出是否符合指定的 Pydantic schema
- **失败处理**：重试该 Worker Agent，传递拒绝原因和改进建议

### 2. 最终结果验证

在 `synthesize_results` 完成后验证：

- **验证目标**：用户的原始需求（`state.prompt`）
- **验证内容**：
  - 综合结果是否完整回答了用户原始问题
  - 是否遗漏了关键信息
  - 整体质量是否达标
- **失败处理**：
  - 重新开始整个流程（从 `analyze_and_plan`）
  - 达到最大迭代次数时，生成失败总结

## 基本用法

### 配置最大迭代次数

```python
from kiva import Kiva

# 全局配置
kiva = Kiva(
    base_url="...",
    api_key="...",
    model="gpt-4o",
    max_iterations=3,  # 全局默认值
)

# Per-agent 配置
@kiva.agent("complex_task", "处理复杂任务", max_iterations=5)
def complex_task(query: str) -> str:
    """处理需要更多重试机会的复杂任务"""
    return f"Result for {query}"

# Per-run 配置
result = kiva.run("执行任务", max_iterations=4)
```

### 优先级

配置优先级从高到低：
1. `kiva.run(max_iterations=...)` - 单次执行配置
2. `@kiva.agent(..., max_iterations=...)` - Agent 级配置
3. `Kiva(max_iterations=...)` - 全局默认配置

## 自定义验证器

使用 `@kiva.verifier` 装饰器定义自定义验证规则：

```python
from kiva import Kiva, VerificationResult, VerificationStatus

kiva = Kiva(base_url="...", api_key="...", model="gpt-4o")

@kiva.verifier("length_check")
def check_length(
    task: str,
    output: str,
    context: dict | None = None
) -> VerificationResult:
    """检查输出长度是否满足最小要求"""
    if len(output) < 50:
        return VerificationResult(
            status=VerificationStatus.FAILED,
            rejection_reason="输出太短",
            improvement_suggestions=["请提供更详细的回答"],
        )
    return VerificationResult(status=VerificationStatus.PASSED)

@kiva.verifier("keyword_check", priority=10)  # 高优先级，先执行
def check_keywords(
    task: str,
    output: str,
    context: dict | None = None
) -> VerificationResult:
    """检查输出是否包含必要关键词"""
    required_keywords = context.get("required_keywords", []) if context else []
    missing = [kw for kw in required_keywords if kw.lower() not in output.lower()]
    
    if missing:
        return VerificationResult(
            status=VerificationStatus.FAILED,
            rejection_reason=f"缺少关键词: {', '.join(missing)}",
            improvement_suggestions=[f"请在回答中包含: {', '.join(missing)}"],
        )
    return VerificationResult(status=VerificationStatus.PASSED)
```

### 验证器参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `str \| None` | 验证器名称，默认使用函数名 |
| `priority` | `int` | 执行优先级，数值越大越先执行，默认 0 |

### 验证器函数签名

```python
def verifier_func(
    task: str,           # 分配给 Worker 的任务
    output: str,         # Worker 的输出
    context: dict | None # 可选的上下文信息
) -> VerificationResult:
    ...
```

## VerificationResult 模型

```python
from kiva import VerificationResult, VerificationStatus

result = VerificationResult(
    status=VerificationStatus.PASSED,  # PASSED, FAILED, SKIPPED
    rejection_reason="输出不完整",      # 失败原因（可选）
    improvement_suggestions=[           # 改进建议列表
        "添加更多细节",
        "包含具体数据",
    ],
    field_errors={                      # Pydantic 字段级错误
        "name": "Field required",
    },
    validator_name="custom_check",      # 验证器名称
    confidence=0.95,                    # 置信度 (0.0-1.0)
)
```

### VerificationStatus 枚举

| 状态 | 说明 |
|------|------|
| `PASSED` | 验证通过 |
| `FAILED` | 验证失败，需要重试 |
| `SKIPPED` | 验证被跳过（如验证器本身出错） |

## Pydantic Schema 验证

可以为 Agent 输出指定 Pydantic schema 进行结构验证：

```python
from pydantic import BaseModel

class WeatherResponse(BaseModel):
    city: str
    temperature: float
    condition: str
    humidity: int

# 在验证时会自动检查输出是否符合 schema
# 如果输出是 JSON 格式，会进行字段级验证
```

## 重试上下文

当验证失败触发重试时，系统会构建完整的重试上下文：

```python
from kiva import RetryContext

context = RetryContext(
    iteration=2,                    # 当前重试次数
    max_iterations=3,               # 最大重试次数
    original_task="分析数据",        # 原始任务
    previous_outputs=[              # 之前的输出
        "第一次尝试的输出",
        "第二次尝试的输出",
    ],
    previous_rejections=[           # 之前的拒绝原因
        VerificationResult(...),
    ],
    task_history=[...],             # 任务历史
)
```

重试时，Agent 会收到包含以下信息的 prompt：
- 原始任务
- 之前的拒绝原因
- 改进建议
- 明确指示尝试不同的方法

## 事件系统

验证过程会发出以下事件：

### Worker 验证事件

| 事件类型 | 说明 |
|----------|------|
| `worker_verification_start` | 开始 Worker 输出验证 |
| `worker_verification_passed` | 所有 Worker 验证通过 |
| `worker_verification_failed` | 一个或多个 Worker 验证失败 |
| `worker_verification_max_reached` | 达到最大 Worker 验证迭代次数 |

### 最终验证事件

| 事件类型 | 说明 |
|----------|------|
| `final_verification_start` | 开始最终结果验证 |
| `final_verification_passed` | 最终结果验证通过 |
| `final_verification_failed` | 最终结果验证失败 |
| `final_verification_max_reached` | 达到最大工作流迭代次数 |

### 重试事件

| 事件类型 | 说明 |
|----------|------|
| `retry_triggered` | 触发 Worker 重试 |
| `retry_completed` | Worker 重试完成 |
| `retry_skipped` | 重试被跳过 |

### 监听事件

```python
async for event in kiva.run_async("任务", console=False):
    if event.type == "worker_verification_failed":
        print(f"Worker 验证失败: {event.data}")
    elif event.type == "retry_triggered":
        print(f"触发重试: 第 {event.data['iteration']} 次")
```

## 优雅降级

系统在以下情况下会优雅降级：

1. **所有重试失败**：返回最后一次输出，附带警告标志
2. **验证器本身出错**：跳过该验证器，继续执行
3. **达到最大迭代次数**：
   - Worker 级别：继续进行结果合成
   - Workflow 级别：返回失败总结，包含原始请求和失败原因

## 完整示例

```python
from kiva import Kiva, VerificationResult, VerificationStatus

# 创建 Kiva 实例
kiva = Kiva(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    model="gpt-4o",
    max_iterations=3,
)

# 自定义验证器：检查输出质量
@kiva.verifier("quality_check", priority=5)
def check_quality(
    task: str,
    output: str,
    context: dict | None = None
) -> VerificationResult:
    """检查输出质量"""
    issues = []
    suggestions = []
    
    # 检查长度
    if len(output) < 100:
        issues.append("输出过短")
        suggestions.append("请提供更详细的回答")
    
    # 检查是否包含具体内容
    if "TODO" in output or "待补充" in output:
        issues.append("包含未完成内容")
        suggestions.append("请完成所有内容，不要留下占位符")
    
    if issues:
        return VerificationResult(
            status=VerificationStatus.FAILED,
            rejection_reason="; ".join(issues),
            improvement_suggestions=suggestions,
        )
    
    return VerificationResult(status=VerificationStatus.PASSED)

# 定义 Agent
@kiva.agent("researcher", "研究和分析主题")
def research(topic: str) -> str:
    """研究指定主题"""
    return f"关于 {topic} 的研究结果..."

@kiva.agent("writer", "撰写内容", max_iterations=5)
def write(content: str) -> str:
    """撰写内容"""
    return f"撰写的内容: {content}"

# 运行
result = kiva.run("研究并撰写一篇关于人工智能的文章")
print(result)
```

## API 参考

### Kiva 类

```python
class Kiva:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_iterations: int = 3,  # 全局最大迭代次数
    ): ...
    
    def agent(
        self,
        name: str,
        description: str,
        max_iterations: int | None = None,  # Per-agent 配置
    ) -> Callable: ...
    
    def verifier(
        self,
        name: str | None = None,
        priority: int = 0,
    ) -> Callable: ...
    
    def run(
        self,
        prompt: str,
        console: bool = True,
        max_iterations: int | None = None,  # Per-run 配置
    ) -> str | None: ...
    
    async def run_async(
        self,
        prompt: str,
        console: bool = True,
        max_iterations: int | None = None,
    ) -> str | None: ...
    
    def get_verifiers(self) -> list[RegisteredVerifier]: ...
```

### VerificationResult

```python
class VerificationResult(BaseModel):
    status: VerificationStatus
    rejection_reason: str | None = None
    improvement_suggestions: list[str] = []
    field_errors: dict[str, str] = {}
    validator_name: str = "default"
    confidence: float = 1.0  # 0.0 - 1.0
```

### RetryContext

```python
class RetryContext(BaseModel):
    iteration: int
    max_iterations: int
    previous_outputs: list[str] = []
    previous_rejections: list[VerificationResult] = []
    task_history: list[dict[str, Any]] = []
    original_task: str
```

## 最佳实践

1. **合理设置迭代次数**：通常 3-5 次足够，过多会增加延迟和成本
2. **验证器保持简单**：复杂逻辑应该在 Agent 中处理
3. **使用优先级**：关键验证器设置高优先级，快速失败
4. **提供有用的建议**：`improvement_suggestions` 应该具体可操作
5. **监控事件**：在生产环境中监控验证事件，了解失败模式
6. **优雅降级**：确保即使验证失败，用户也能得到有用的反馈
