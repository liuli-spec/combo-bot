"""DeepSeek AI 集成模块。

为 combo_bot 提供 AI 驱动的分析能力：

* 回测结果深度分析
* 市场行情解读
* 策略参数建议
* 风险诊断

使用方式::

    from combo_bot.deepseek import DeepSeekClient

    client = DeepSeekClient(api_key="sk-xxx")  # 或从环境变量 DEEPSEEK_API_KEY 读取
    analysis = await client.analyze_backtest(backtest_result)

API key 配置优先级：
    1. 直接传入 ``api_key`` 参数
    2. 环境变量 ``DEEPSEEK_API_KEY``
    3. 工作目录下的 ``.env`` 文件中的 ``DEEPSEEK_API_KEY``
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

logger = logging.getLogger(__name__)

# Try to load .env file (best-effort, don't error if python-dotenv isn't installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── configuration ──────────────────────────────────────────────────

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_REASONING_MODEL = "deepseek-reasoner"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.3  # 分析类任务用低温


def _default_api_key() -> str:
    """从环境变量或 .env 文件读取 API key。"""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        logger.warning(
            "DEEPSEEK_API_KEY 未设置 —— DeepSeek 功能不可用。"
            "请在环境变量或 .env 文件中设置 DEEPSEEK_API_KEY。"
        )
    return key


# ── prompt templates ────────────────────────────────────────────────

BACKTEST_ANALYSIS_SYSTEM = """你是一位专业的量化交易分析师，专注于加密货币永续合约的网格+趋势融合策略分析。

你的任务是分析回测结果，给出专业、客观、结构化的分析报告。请始终：

1. 用简体中文回复
2. 用数据和指标说话，不做空洞的评价
3. 指出策略的优点和风险
4. 给出可操作的改进建议
5. 语言专业但不晦涩

分析框架：
- 综合表现：ADG（日均收益）、总收益、最大回撤、收益回撤比
- 风险指标：Sharpe（夏普比率）、Sortino（索提诺比率）、Calmar（卡玛比率）
- 交易质量：胜率、成交笔数、手续费占比
- 组件分解：Grid 盈亏 vs Trend 盈亏各自贡献
- 改进建议：参数调整方向、风险控制建议"""

MARKET_ANALYSIS_SYSTEM = """你是一位加密货币市场分析师，专注于永续合约市场。

根据提供的行情数据摘要（价格、成交量、EMA趋势、RSI、MACD、布林带等指标），给出市场分析：

1. 当前趋势判断（强牛/牛/中性/熊/强熊）
2. 关键支撑与阻力位
3. 波动率评估
4. 风险提示
5. 与该策略风格（网格+趋势融合）的适配度评估

用简体中文回复，保持专业和客观。"""

STRATEGY_ADVISOR_SYSTEM = """你是一位量化策略顾问，专门帮助优化 combo-futures 的网格+趋势融合策略参数。

你将收到当前策略配置和回测结果。请：

1. 分析当前参数设置的合理性
2. 指出可能过拟合的风险
3. 建议参数调整方向和幅度
4. 推荐值得尝试的参数范围
5. 评估策略在不同市场环境下的适应性

用简体中文回复，给出具体、可操作的数值建议。"""


# ── client ─────────────────────────────────────────────────────────

@dataclass
class DeepSeekConfig:
    """DeepSeek API 配置。"""

    api_key: str = field(default_factory=_default_api_key)
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


class DeepSeekClient:
    """DeepSeek API 异步客户端。

    封装聊天补全、流式响应、以及 combo_bot 专用的分析任务。"""

    def __init__(self, config: DeepSeekConfig | None = None, api_key: str | None = None):
        self.config = config or DeepSeekConfig()
        if api_key:
            self.config.api_key = api_key
        self._client = None
        self._initialized = False

    @property
    def available(self) -> bool:
        """DeepSeek API 是否可用（已配置 API key）。"""
        return bool(self.config.api_key)

    async def _ensure_client(self):
        """延迟初始化 httpx 异步客户端。"""
        if self._initialized:
            return
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx 未安装。请运行: pip install httpx"
            )
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=httpx.Timeout(self.config.timeout_seconds),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        self._initialized = True

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncIterator[str]:
        """发送聊天请求。

        Args:
            messages: 消息列表 [{"role": "system/user/assistant", "content": "..."}]
            model: 模型名称，默认使用 config 中的设置
            max_tokens: 最大 token 数
            temperature: 温度参数
            stream: 是否流式返回

        Returns:
            非流式: dict（完整响应）
            流式: AsyncIterator[str]（增量内容）
        """
        await self._ensure_client()
        if self._client is None:
            raise RuntimeError("客户端初始化失败")

        body: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature or self.config.temperature,
            "stream": stream,
        }

        if stream:
            return self._stream_chat(body)
        else:
            return await self._chat(body)

    async def _chat(self, body: dict[str, Any]) -> dict[str, Any]:
        """非流式聊天请求。"""
        assert self._client is not None
        try:
            resp = await self._client.post("/chat/completions", json=body)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("DeepSeek API 请求失败: %s", exc)
            raise

    async def _stream_chat(
        self, body: dict[str, Any]
    ) -> AsyncIterator[str]:
        """流式聊天请求，逐 chunk yield 内容增量。"""
        assert self._client is not None
        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=body
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = (
                                data.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if delta:
                                yield delta
                        except json.JSONDecodeError:
                            continue
        except Exception as exc:
            logger.error("DeepSeek 流式请求失败: %s", exc)
            raise

    async def close(self) -> None:
        """关闭 HTTP 客户端。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._initialized = False

    # ── 业务方法 ─────────────────────────────────────────────────

    async def analyze_backtest(
        self,
        result: dict[str, Any],
        *,
        config_summary: str = "",
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """分析回测结果。

        Args:
            result: 回测结果字典，含 final_balance, total_pnl, max_drawdown,
                    sharpe_ratio, sortino_ratio, calmar_ratio, win_rate,
                    adg, n_trades, grid_pnl, trend_pnl 等
            config_summary: 策略配置简述（可选，帮助 AI 理解上下文）
            stream: 是否流式返回

        Returns:
            AI 分析文本
        """
        if not self.available:
            return "错误: DeepSeek API key 未配置。请设置环境变量 DEEPSEEK_API_KEY。"

        # 格式化回测结果
        result_text = _format_backtest_for_prompt(result, config_summary)

        messages = [
            {"role": "system", "content": BACKTEST_ANALYSIS_SYSTEM},
            {
                "role": "user",
                "content": f"请分析以下回测结果，给出专业评估和改进建议：\n\n{result_text}",
            },
        ]

        if stream:
            return self._stream_to_text(self.chat(messages, stream=True))
        else:
            resp = await self.chat(messages)
            try:
                return resp["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                logger.error("DeepSeek 响应格式异常: %s", resp)
                return "错误: API 返回了异常格式，请检查日志。"

    async def analyze_market(
        self,
        market_data: dict[str, Any],
        *,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """分析市场行情。

        Args:
            market_data: 市场数据摘要，含价格、指标等
            stream: 是否流式返回

        Returns:
            AI 市场分析文本
        """
        if not self.available:
            return "错误: DeepSeek API key 未配置。"

        data_text = _format_market_for_prompt(market_data)

        messages = [
            {"role": "system", "content": MARKET_ANALYSIS_SYSTEM},
            {
                "role": "user",
                "content": f"请分析以下市场数据：\n\n{data_text}",
            },
        ]

        if stream:
            return self._stream_to_text(self.chat(messages, stream=True))
        else:
            resp = await self.chat(messages)
            return resp["choices"][0]["message"]["content"]

    async def strategy_advisor(
        self,
        params: dict[str, Any],
        backtest: dict[str, Any],
        *,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """策略参数顾问。

        Args:
            params: 当前策略参数
            backtest: 回测结果
            stream: 是否流式返回

        Returns:
            AI 策略建议
        """
        if not self.available:
            return "错误: DeepSeek API key 未配置。"

        context = f"""当前策略参数：
```json
{json.dumps(params, indent=2, ensure_ascii=False)}
```

回测结果：
```json
{json.dumps(backtest, indent=2, ensure_ascii=False, default=str)}
```"""

        messages = [
            {"role": "system", "content": STRATEGY_ADVISOR_SYSTEM},
            {"role": "user", "content": f"请分析并给出优化建议：\n\n{context}"},
        ]

        if stream:
            return self._stream_to_text(self.chat(messages, stream=True))
        else:
            resp = await self.chat(messages)
            return resp["choices"][0]["message"]["content"]

    async def freeform_chat(
        self,
        prompt: str,
        *,
        context: str = "",
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """自由对话（带上回测/市场上下文）。

        Args:
            prompt: 用户问题
            context: 可选的上下文（回测结果、市场数据等）
            stream: 是否流式返回

        Returns:
            AI 回复
        """
        if not self.available:
            return "错误: DeepSeek API key 未配置。"

        messages = [{"role": "system", "content": BACKTEST_ANALYSIS_SYSTEM}]
        if context:
            messages.append({
                "role": "user",
                "content": f"以下是上下文数据供参考：\n{context}",
            })
            messages.append({
                "role": "assistant",
                "content": "已了解数据背景，请提问。",
            })
        messages.append({"role": "user", "content": prompt})

        if stream:
            return self._stream_to_text(self.chat(messages, stream=True))
        else:
            resp = await self.chat(messages)
            return resp["choices"][0]["message"]["content"]

    async def _stream_to_text(
        self, stream: AsyncIterator[str]
    ) -> AsyncIterator[str]:
        """将流式 chat 结果直接透传为文本片段。"""
        async for chunk in stream:
            yield chunk


# ── helper functions ───────────────────────────────────────────────

def _format_backtest_for_prompt(
    result: dict[str, Any], config_summary: str = ""
) -> str:
    """将回测结果字典格式化为适合 prompt 的文本。"""
    lines = []

    if config_summary:
        lines.append(f"策略配置: {config_summary}")
        lines.append("")

    metrics: dict[str, tuple[str, str]] = {
        "final_balance": ("最终余额", "$"),
        "total_pnl": ("总盈亏", "$"),
        "total_fees": ("总手续费", "$"),
        "n_trades": ("成交笔数", ""),
        "win_rate": ("胜率", "%"),
        "adg": ("日均收益 (ADG)", "%"),
        "max_drawdown": ("最大回撤", "%"),
        "sharpe_ratio": ("夏普比率", ""),
        "sortino_ratio": ("索提诺比率", ""),
        "calmar_ratio": ("卡玛比率", ""),
        "grid_pnl": ("Grid 盈亏", "$"),
        "trend_pnl": ("Trend 盈亏", "$"),
        "duration_days": ("回测天数", " 天"),
    }

    for key, (label, suffix) in metrics.items():
        val = result.get(key)
        if val is not None:
            if suffix in ("%",):
                lines.append(f"- {label}: {float(val) * 100:.2f}%")
            elif suffix == "$":
                lines.append(f"- {label}: ${float(val):,.2f}")
            elif suffix == " 天":
                lines.append(f"- {label}: {float(val):.1f} 天")
            else:
                lines.append(f"- {label}: {val}")

    return "\n".join(lines)


def _format_market_for_prompt(data: dict[str, Any]) -> str:
    """将市场数据格式化为 prompt 文本。"""
    lines = []

    for symbol, info in data.items():
        if not isinstance(info, dict):
            continue
        lines.append(f"## {symbol}")
        lines.append(f"- 最新价: ${info.get('last', 'N/A')}")
        lines.append(f"- 24h 涨跌: {info.get('change_24h', 'N/A')}")
        lines.append(f"- 成交量: {info.get('volume', 'N/A')}")
        lines.append(f"- EMA 趋势: {info.get('ema_trend', 'N/A')}")
        lines.append(f"- RSI: {info.get('rsi', 'N/A')}")
        lines.append(f"- MACD 信号: {info.get('macd_signal', 'N/A')}")
        regime = info.get("regime")
        if regime:
            lines.append(f"- 行情模式: {regime}")
        lines.append("")

    return "\n".join(lines)


# ── 同步简易工具 ──────────────────────────────────────────────────

def quick_test(api_key: str | None = None, model: str | None = None) -> dict[str, Any]:
    """快速测试 DeepSeek API 连通性（同步封装）。

    Returns:
        {"ok": True/False, "message": "...", "model": "...", "usage": {...}}
    """
    import asyncio

    async def _test():
        key = api_key or _default_api_key()
        if not key:
            return {"ok": False, "message": "未配置 API key"}

        client = DeepSeekClient(api_key=key)
        if model:
            client.config.model = model

        try:
            resp = await client.chat(
                [{"role": "user", "content": "请回复'OK'，仅回复这两个字母。"}],
                max_tokens=10,
            )
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = resp.get("usage", {})
            return {
                "ok": True,
                "message": f"连通成功，回复: {content.strip()}",
                "model": resp.get("model", ""),
                "usage": usage,
            }
        except Exception as exc:
            return {"ok": False, "message": str(exc)}
        finally:
            await client.close()

    return asyncio.run(_test())
