from __future__ import annotations

import asyncio
import time

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register

from .hr_ble import HeartRateMonitor, ScannedDevice, scan_devices

ALERT_KINDS = ("high", "low")


@register(
    "astrbot_plugin_heart_rate",
    "baiye",
    "通过蓝牙收取标准心率设备的心率数据，并在心率超过阈值时主动发送问候消息。",
    "1.0.0",
)
class HeartRatePlugin(Star):
    """Bluetooth heart rate monitoring and threshold-based proactive alerts."""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = config
        self.monitor = HeartRateMonitor()
        self._targets: list[str] = []
        self._episodes = {
            "high": {"active": False, "last_sent": 0.0},
            "low": {"active": False, "last_sent": 0.0},
        }
        self._alert_task: asyncio.Task | None = None
        # Serializes LLM generation so overlapping alert evaluations cannot
        # fire multiple model requests at the same time.
        self._llm_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Load bound conversations and optionally auto-start monitoring."""
        try:
            stored = await self.get_kv_data("bound_targets", [])
            if isinstance(stored, list):
                self._targets = [t for t in stored if isinstance(t, str)]
        except Exception as e:
            logger.warning(f"Failed to load bound heart rate targets: {e!s}")

        if self.config.get("auto_start", False):
            asyncio.create_task(self._auto_start())

    async def _auto_start(self) -> None:
        """Start the collector after AstrBot finishes booting."""
        await asyncio.sleep(3)
        try:
            await self.monitor.start(self._build_settings(""))
            self._ensure_alert_task()
            logger.info("Heart rate monitoring auto-started.")
        except Exception as e:
            logger.error(f"Failed to auto-start heart rate monitoring: {e!s}")

    async def terminate(self) -> None:
        """Release the Bluetooth radio when the plugin is unloaded."""
        if self._alert_task:
            self._alert_task.cancel()
            self._alert_task = None
        await self.monitor.stop()

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @filter.command("hr")
    async def hr_status(self, event: AstrMessageEvent):
        """查看心率监测状态、最新心率与当前阈值"""
        st = self.monitor.status()
        if st["running"]:
            mode_text = "广播被动接收" if st["mode"] == "advertise" else "GATT 连接"
            link_text = (
                ("正在监听广播" if st["mode"] == "advertise" else "已连接")
                if (st["connected"] or st["scanning"])
                else "连接中/等待设备"
            )
            lines = [
                "🫀 心率监测运行中",
                f"模式：{mode_text}（{link_text}）",
                f"设备：{st['device_name'] or st['device_address'] or '未知'}",
            ]
            hr = st["latest_hr"]
            if hr is not None:
                age = time.time() - st["last_hr_ts"]
                fresh = age <= int(self.config.get("data_timeout", 15))
                lines.append(
                    f"最新心率：{hr} bpm（{int(age)} 秒前）"
                    + ("" if fresh else "，数据可能已中断")
                )
            else:
                lines.append("最新心率：暂未收到数据")
            if st["last_error"]:
                lines.append(f"最近错误：{st['last_error']}")
        else:
            lines = ["🫀 心率监测未运行，发送 /hr_start 开始监测"]

        high = self.config.get("high_threshold", 100)
        low = self.config.get("low_threshold", 0)
        lines.append(
            f"阈值：高于 {high} bpm 提醒" + (f"，低于 {low} bpm 提醒" if low else "")
        )
        lines.append(
            "问候方式："
            + (
                "🤖 AI 按人设生成"
                if self.config.get("use_llm", False)
                else "📝 固定模板"
            )
        )
        lines.append(f"提醒目标会话：{len(self._targets)} 个（/hr_bind 绑定当前会话）")
        yield event.plain_result("\n".join(lines))

    @filter.command("hr_scan")
    async def hr_scan(self, event: AstrMessageEvent, duration: int = 0):
        """扫描附近的蓝牙心率设备，可带扫描秒数，如 /hr_scan 10"""
        if self.monitor.running:
            yield event.plain_result(
                "监测运行中无法执行扫描，请先发送 /hr_stop 停止监测后再试。"
            )
            return

        timeout = duration if duration > 0 else int(self.config.get("scan_timeout", 12))
        timeout = min(max(timeout, 3), 60)
        yield event.plain_result(f"🔍 正在扫描附近蓝牙设备（{timeout} 秒）...")

        try:
            devices = await scan_devices(timeout)
        except ImportError:
            yield event.plain_result(
                "缺少蓝牙依赖 bleak，请在插件目录的 requirements.txt 安装依赖后重启 AstrBot。"
            )
            return
        except Exception as e:
            yield event.plain_result(f"扫描失败：{e!s}\n请确认本机蓝牙已开启。")
            return

        if not devices:
            yield event.plain_result("未发现任何蓝牙设备，请确认设备已开机并靠近电脑。")
            return

        lines = ["扫描完成，发现以下设备："]
        for i, dev in enumerate(devices[:15], 1):
            tag = self._device_tag(dev)
            lines.append(
                f"{i}. {dev.name or '（未命名设备）'} [{dev.address}] "
                f"RSSI {dev.rssi}{tag}"
            )
        lines.append(
            "提示：可在插件配置中填写设备地址/名称，或发送 /hr_start <地址> 直接开始。"
        )
        yield event.plain_result("\n".join(lines))

    @filter.command("hr_start")
    async def hr_start(self, event: AstrMessageEvent, address: str = ""):
        """开始心率监测，可指定设备地址，如 /hr_start AA:BB:CC:DD:EE:FF"""
        if self.monitor.running:
            yield event.plain_result("心率监测已在运行中，无需重复启动。")
            return

        try:
            await self.monitor.start(self._build_settings(address))
        except ImportError:
            yield event.plain_result(
                "缺少蓝牙依赖 bleak，请在插件目录的 requirements.txt 安装依赖后重启 AstrBot。"
            )
            return
        except Exception as e:
            yield event.plain_result(f"启动失败：{e!s}")
            return

        self._reset_episodes()
        self._ensure_alert_task()
        mode = self.config.get("mode", "gatt")
        mode_text = "广播被动接收" if mode == "advertise" else "GATT 连接"
        yield event.plain_result(
            f"✅ 心率监测已启动（{mode_text}）。\n"
            "正在搜索并连接设备，请佩戴好心率设备并稍候，发送 /hr 查看状态。"
        )

    @filter.command("hr_stop")
    async def hr_stop(self, event: AstrMessageEvent):
        """停止心率监测并断开蓝牙连接"""
        if not self.monitor.running:
            yield event.plain_result("心率监测当前未运行。")
            return
        await self.monitor.stop()
        if self._alert_task:
            self._alert_task.cancel()
            self._alert_task = None
        self._reset_episodes()
        yield event.plain_result("🛑 心率监测已停止，蓝牙连接已断开。")

    @filter.command("hr_bind")
    async def hr_bind(self, event: AstrMessageEvent):
        """将当前聊天会话绑定为心率告警接收目标"""
        umo = event.unified_msg_origin
        if umo in self._targets:
            yield event.plain_result("当前会话已经是心率告警目标，无需重复绑定。")
            return
        self._targets.append(umo)
        await self._save_targets()
        yield event.plain_result(
            "✅ 已绑定当前会话。心率超过阈值时，机器人会主动向这里发送问候消息。\n"
            "可用 /hr_unbind 解除绑定。"
        )

    @filter.command("hr_unbind")
    async def hr_unbind(self, event: AstrMessageEvent):
        """解除当前聊天会话的心率告警绑定"""
        umo = event.unified_msg_origin
        if umo not in self._targets:
            yield event.plain_result("当前会话未绑定心率告警。")
            return
        self._targets.remove(umo)
        await self._save_targets()
        yield event.plain_result("已解除当前会话的心率告警绑定。")

    @filter.command("hr_threshold")
    async def hr_threshold(self, event: AstrMessageEvent, high: int, low: int = -1):
        """设置心率阈值，如 /hr_threshold 100 或 /hr_threshold 100 50（0 表示关闭对应提醒）"""
        if not (high == 0 or 30 <= high <= 250):
            yield event.plain_result(
                "过高阈值需在 30-250 bpm 之间，或填 0 关闭过高提醒。"
            )
            return
        new_low = self.config.get("low_threshold", 0) if low == -1 else low
        if not (new_low == 0 or 30 <= new_low <= 250):
            yield event.plain_result(
                "过低阈值需在 30-250 bpm 之间，或填 0 关闭过低提醒。"
            )
            return
        if high and new_low and high <= new_low:
            yield event.plain_result("过高阈值必须大于过低阈值。")
            return

        self.config["high_threshold"] = high
        self.config["low_threshold"] = new_low
        self.config.save_config()
        self._reset_episodes()
        yield event.plain_result(
            f"✅ 阈值已更新：高于 {high or '（关闭）'} bpm 提醒"
            f"，低于 {new_low or '（关闭）'} bpm 提醒。"
        )

    @filter.command("hr_test")
    async def hr_test(self, event: AstrMessageEvent, hr: int):
        """模拟一条心率数据测试告警链路，如 /hr_test 120"""
        if not (30 <= hr <= 250):
            yield event.plain_result("请输入 30-250 bpm 之间的模拟心率。")
            return
        if not self._targets:
            yield event.plain_result(
                "当前没有绑定任何告警会话，请先发送 /hr_bind 绑定后再测试。"
            )
            return
        self.monitor.latest_hr = hr
        self.monitor.last_hr_ts = time.time()
        await self._evaluate_alerts(hr, self.monitor.last_hr_ts)
        mode_text = "AI 按人设生成" if self.config.get("use_llm", False) else "固定模板"
        yield event.plain_result(
            f"已注入模拟心率 {hr} bpm，如越过阈值将以「{mode_text}」方式向 "
            f"{len(self._targets)} 个绑定会话发送提醒。"
        )

    # ------------------------------------------------------------------
    # Alert engine
    # ------------------------------------------------------------------

    def _build_settings(self, address_override: str) -> dict:
        """Assemble BLE settings from plugin config for the monitor worker."""
        return {
            "mode": self.config.get("mode", "gatt"),
            "device_address": address_override or self.config.get("device_address", ""),
            "device_name": self.config.get("device_name", ""),
            "scan_timeout": int(self.config.get("scan_timeout", 12)),
            "reconnect_interval": int(self.config.get("reconnect_interval", 5)),
        }

    def _ensure_alert_task(self) -> None:
        """Start the periodic alert evaluation task if it is not running."""
        if self._alert_task is None or self._alert_task.done():
            self._alert_task = asyncio.create_task(self._alert_loop())

    def _reset_episodes(self) -> None:
        """Clear active alert episodes (used on start/stop/threshold change)."""
        for kind in ALERT_KINDS:
            self._episodes[kind]["active"] = False
            self._episodes[kind]["last_sent"] = 0.0

    async def _save_targets(self) -> None:
        """Persist bound conversation UMOs to the plugin KV store."""
        try:
            await self.put_kv_data("bound_targets", self._targets)
        except Exception as e:
            logger.warning(f"Failed to save heart rate targets: {e!s}")

    async def _alert_loop(self) -> None:
        """Periodically evaluate the latest heart rate against thresholds."""
        while self.monitor.running:
            try:
                await asyncio.sleep(2)
                hr = self.monitor.latest_hr
                ts = self.monitor.last_hr_ts
                if hr is None:
                    continue
                if time.time() - ts > int(self.config.get("data_timeout", 15)):
                    # Stale data (device out of range): reset episodes so a
                    # later episode is alerted as a new one.
                    self._reset_episodes()
                    continue
                await self._evaluate_alerts(hr, ts)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # A transient error in one evaluation (e.g. a provider hiccup)
                # must not kill the loop, otherwise alerts stay silent until
                # the plugin is reloaded.
                logger.warning(f"Heart rate alert iteration error: {e!s}")

    async def _evaluate_alerts(self, hr: int, _ts: float) -> None:
        """Check one heart rate sample against both thresholds.

        Args:
            hr: Current heart rate in bpm.
            _ts: Timestamp of the sample (kept for future extensions).
        """
        high = int(self.config.get("high_threshold", 0))
        low = int(self.config.get("low_threshold", 0))
        margin = max(int(self.config.get("recovery_margin", 5)), 0)

        if high:
            await self._evaluate_episode(
                kind="high",
                triggered=hr > high,
                recovered=hr <= high - margin,
                hr=hr,
                threshold=high,
            )
        if low:
            await self._evaluate_episode(
                kind="low",
                triggered=hr < low,
                recovered=hr >= low + margin,
                hr=hr,
                threshold=low,
            )

    async def _evaluate_episode(
        self,
        *,
        kind: str,
        triggered: bool,
        recovered: bool,
        hr: int,
        threshold: int,
    ) -> None:
        """Manage one threshold episode and dispatch alert messages.

        Args:
            kind: Episode kind ("high" or "low").
            triggered: Whether the current sample crosses the threshold.
            recovered: Whether the sample returned past the recovery margin.
            hr: Current heart rate in bpm.
            threshold: Triggered threshold in bpm.
        """
        episode = self._episodes[kind]
        now = time.time()

        if triggered:
            should_send = False
            if not episode["active"]:
                episode["active"] = True
                should_send = True
            else:
                cooldown = int(self.config.get("alert_cooldown", 60))
                if cooldown > 0 and now - episode["last_sent"] >= cooldown:
                    should_send = True
            if should_send:
                episode["last_sent"] = now
                text = await self._resolve_alert_text(kind, hr, threshold)
                await self._broadcast(text)
        elif episode["active"] and recovered:
            episode["active"] = False
            if self.config.get("recovery_enabled", True):
                text = await self._resolve_alert_text("recovery", hr, threshold)
                await self._broadcast(text)

    async def _resolve_alert_text(self, scene: str, hr: int, threshold: int) -> str:
        """Resolve the final message text for an alert scene.

        When AI generation is enabled, asks the current chat provider to
        produce a persona-consistent greeting; falls back to the configured
        fixed template when the model is unavailable or generation fails.

        Args:
            scene: Alert scene: "high", "low" or "recovery".
            hr: Current heart rate in bpm.
            threshold: Triggered threshold in bpm.

        Returns:
            The message text ready to be sent.
        """
        ai_prompt_key = {
            "high": "llm_high_prompt",
            "low": "llm_low_prompt",
            "recovery": "llm_recovery_prompt",
        }.get(scene, "llm_high_prompt")
        fallback_key = {
            "high": "high_message",
            "low": "low_message",
            "recovery": "recovery_message",
        }.get(scene, "high_message")

        if self.config.get("use_llm", False):
            task_prompt = self._format_message(
                self.config.get(ai_prompt_key, ""), hr, threshold
            )
            text = await self._generate_ai_message(task_prompt)
            if text:
                return text
            logger.warning(
                "AI greeting generation failed or returned empty, "
                "falling back to fixed template."
            )

        return self._format_message(self.config.get(fallback_key, ""), hr, threshold)

    async def _generate_ai_message(self, task_prompt: str) -> str:
        """Ask the current chat provider to generate a proactive greeting.

        The default persona's system prompt is used so the output matches
        the configured character, mirroring how AstrBot future tasks wake
        the agent. Generation is serialized with a lock and has no chat
        history: each greeting is an independent one-shot completion.

        Args:
            task_prompt: Formatted task instruction describing the event.

        Returns:
            Generated message text, or an empty string on any failure.
        """
        if not task_prompt:
            return ""
        async with self._llm_lock:
            # Build an ordered candidate list: the provider the user is
            # currently chatting with first, then every other configured chat
            # provider. This avoids getting stuck on the first provider in
            # config (which may be unreachable, e.g. 404) when the user's
            # working model is selected via per-session routing in abconf.
            candidates: list = []
            curr = getattr(self.context.provider_manager, "curr_provider_inst", None)
            if curr is not None:
                candidates.append(curr)
            for p in self.context.get_all_providers() or []:
                if p not in candidates:
                    candidates.append(p)
            if not candidates:
                logger.warning(
                    "No chat provider configured for AI heart rate greeting."
                )
                return ""

            system_prompt = ""
            try:
                persona = self.context.persona_manager.selected_default_persona
                if persona is not None:
                    system_prompt = persona.system_prompt or ""
            except Exception as e:
                logger.debug(f"Failed to read default persona prompt: {e!s}")

            extra = (self.config.get("llm_system_extra") or "").strip()
            if extra:
                system_prompt = f"{system_prompt}\n\n{extra}".strip()

            last_error: Exception | None = None
            for provider in candidates:
                try:
                    pid = getattr(provider, "provider_config", {}).get("id", "?")
                    model = (
                        provider.get_model() if hasattr(provider, "get_model") else "?"
                    )
                    logger.debug(
                        "AI heart rate greeting trying provider: id=%s model=%s",
                        pid,
                        model,
                    )
                    resp = await provider.text_chat(
                        prompt=task_prompt,
                        system_prompt=system_prompt or None,
                        session_id="astrbot_plugin_heart_rate",
                    )
                    text = (getattr(resp, "completion_text", "") or "").strip()
                    if text:
                        return text
                    logger.warning(
                        "Provider %s returned empty greeting, trying next.", pid
                    )
                except Exception as e:
                    last_error = e
                    pid = getattr(provider, "provider_config", {}).get("id", "?")
                    logger.warning("AI greeting failed on provider %s: %s", pid, str(e))
            if last_error is not None:
                logger.warning(
                    "All chat providers failed for AI heart rate greeting. "
                    f"Last error: {last_error!s}"
                )
            return ""

    def _format_message(self, template: str, hr: int, threshold: int) -> str:
        """Render an alert template with safe placeholder fallbacks.

        Args:
            template: Message template containing optional placeholders.
            hr: Current heart rate in bpm.
            threshold: Triggered threshold in bpm.

        Returns:
            The formatted message, or a plain fallback on template errors.
        """
        if not template:
            return ""
        try:
            return template.format(
                hr=hr,
                threshold=threshold,
                device=self.monitor.device_name
                or self.monitor.device_address
                or "心率设备",
            )
        except Exception:
            return f"心率提醒：当前 {hr} bpm，阈值 {threshold} bpm。"

    async def _broadcast(self, text: str) -> None:
        """Send a proactive message to every bound conversation.

        Args:
            text: Plain text message to send.
        """
        if not text or not self._targets:
            return
        chain = MessageChain().message(text)
        for umo in list(self._targets):
            try:
                await self.context.send_message(umo, chain)
            except Exception as e:
                logger.warning(f"Failed to send heart rate alert to {umo}: {e!s}")

    @staticmethod
    def _device_tag(dev: ScannedDevice) -> str:
        """Build a human-readable capability tag for a scanned device."""
        tags = []
        if dev.has_hr_service:
            tags.append("标准心率服务")
        if dev.advertised_hr is not None:
            tags.append(f"广播心率 {dev.advertised_hr} bpm")
        return f"（{'，'.join(tags)}）" if tags else "（未发现心率服务）"
