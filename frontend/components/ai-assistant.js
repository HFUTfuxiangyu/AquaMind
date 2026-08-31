(function () {
    'use strict';

    const API_BASE = 'http://localhost:5000';
    const STORAGE_KEY = 'aquamind_ai_messages';
    const messages = [];
    let elements = {};
    let sending = false;

    function escapeHtml(value) {
        return String(value == null ? '' : value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function getPageTitle() {
        const h1 = document.querySelector('h1');
        if (h1 && h1.textContent.trim()) return h1.textContent.trim();
        return document.title || '智慧水务系统';
    }

    function getDataContext() {
        const aqua = window.AquaMindData;
        if (!aqua) return '当前页面没有检测到共享数据。';

        const meta = typeof aqua.getMeta === 'function' ? aqua.getMeta() : aqua.meta;
        const data = typeof aqua.getData === 'function' ? aqua.getData() : aqua.data;
        const columns = typeof aqua.getColumns === 'function' ? aqua.getColumns() : aqua.columns;
        const count = Array.isArray(data) ? data.length : 0;
        const sample = Array.isArray(data) && data.length ? data.slice(0, 3) : [];

        return [
            `页面: ${getPageTitle()}`,
            `记录数: ${count}`,
            `字段: ${Array.isArray(columns) ? columns.join(', ') : '未知'}`,
            `数据源: ${(meta && (meta.source || meta.dataSource || meta.mode)) || '未知'}`,
            `样例: ${JSON.stringify(sample).slice(0, 1200)}`
        ].join('\n');
    }

    function getSystemPrompt() {
        return [
            '你是 AquaMind Pro 智慧水务系统的专业助手。',
            '回答必须围绕当前页面、共享数据和水务业务场景，优先给出可执行建议。',
            '如果数据不足，要明确说明缺失项，不要编造实时数据。',
            '',
            getDataContext()
        ].join('\n');
    }

    function loadMessages() {
        try {
            const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
            if (Array.isArray(saved)) messages.push(...saved.slice(-20));
        } catch (error) {
            messages.length = 0;
        }
        if (!messages.length) {
            messages.push({
                role: 'assistant',
                content: '你好，我是 AquaMind 智慧水务助手。可以帮你分析水质、加药、能耗、设备健康和数据异常。'
            });
        }
    }

    function saveMessages() {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(messages.slice(-20)));
        } catch (error) {
            // localStorage may be unavailable in restricted environments.
        }
    }

    function createShell() {
        if (document.querySelector('.aqua-ai-assistant')) return;
        document.body.classList.add('aqua-unified-ai');

        const root = document.createElement('div');
        root.className = 'aqua-ai-assistant';
        root.innerHTML = [
            '<button class="aqua-ai-toggle" type="button" title="AI 助手" aria-label="AI 助手">',
            '  <i class="fas fa-robot"></i>',
            '</button>',
            '<section class="aqua-ai-panel" aria-label="AI 助手面板">',
            '  <header class="aqua-ai-header">',
            '    <div>',
            '      <strong>AI 助手</strong>',
            '      <span>水务运行分析</span>',
            '    </div>',
            '    <button class="aqua-ai-close" type="button" title="关闭" aria-label="关闭"><i class="fas fa-times"></i></button>',
            '  </header>',
            '  <div class="aqua-ai-quick">',
            '    <button type="button" data-question="请概括当前页面的关键运行状态。">运行概览</button>',
            '    <button type="button" data-question="当前数据可能有哪些异常或风险？">异常风险</button>',
            '    <button type="button" data-question="给出下一步优化建议。">优化建议</button>',
            '  </div>',
            '  <div class="aqua-ai-messages" role="log" aria-live="polite"></div>',
            '  <form class="aqua-ai-form">',
            '    <textarea class="aqua-ai-input" rows="2" placeholder="输入你的问题..."></textarea>',
            '    <button class="aqua-ai-send" type="submit" title="发送" aria-label="发送"><i class="fas fa-paper-plane"></i></button>',
            '  </form>',
            '</section>'
        ].join('');
        document.body.appendChild(root);

        elements = {
            root,
            toggle: root.querySelector('.aqua-ai-toggle'),
            panel: root.querySelector('.aqua-ai-panel'),
            close: root.querySelector('.aqua-ai-close'),
            messages: root.querySelector('.aqua-ai-messages'),
            form: root.querySelector('.aqua-ai-form'),
            input: root.querySelector('.aqua-ai-input'),
            send: root.querySelector('.aqua-ai-send'),
            quick: root.querySelector('.aqua-ai-quick')
        };
    }

    function renderMessages() {
        elements.messages.innerHTML = messages.map((message) => {
            const roleClass = message.role === 'user' ? 'is-user' : 'is-assistant';
            const name = message.role === 'user' ? '我' : 'AI';
            return [
                `<article class="aqua-ai-message ${roleClass}">`,
                `  <span>${name}</span>`,
                `  <div>${escapeHtml(message.content).replace(/\n/g, '<br>')}</div>`,
                '</article>'
            ].join('');
        }).join('');
        elements.messages.scrollTop = elements.messages.scrollHeight;
    }

    function setOpen(open) {
        elements.root.classList.toggle('is-open', open);
        if (open) setTimeout(() => elements.input.focus(), 0);
    }

    function setSending(value) {
        sending = value;
        elements.send.disabled = value;
        elements.input.disabled = value;
        elements.send.innerHTML = value ? '<i class="fas fa-spinner fa-spin"></i>' : '<i class="fas fa-paper-plane"></i>';
    }

    function extractAssistantText(data) {
        if (typeof data === 'string') return data.trim();
        if (!data || typeof data !== 'object') return '';

        const direct = [data.response, data.reply, data.answer, data.message, data.content, data.output_text]
            .find((value) => typeof value === 'string' && value.trim());
        if (direct) return direct.trim();

        const choiceContent = data.choices && data.choices[0] && data.choices[0].message
            ? data.choices[0].message.content
            : data.choices && data.choices[0] ? data.choices[0].text : '';
        if (typeof choiceContent === 'string' && choiceContent.trim()) return choiceContent.trim();
        if (Array.isArray(choiceContent)) {
            const text = choiceContent.map((item) => item && (item.text || item.content || item.value)).filter(Boolean).join('\n');
            if (text.trim()) return text.trim();
        }

        if (Array.isArray(data.output)) {
            const text = data.output.flatMap((item) => Array.isArray(item && item.content) ? item.content : [])
                .map((item) => item && (item.text || item.value || item.content)).filter(Boolean).join('\n');
            if (text.trim()) return text.trim();
        }
        return '';
    }

    function finiteNumber(row, column) {
        if (!row || !column) return null;
        const value = Number.parseFloat(row[column]);
        return Number.isFinite(value) ? value : null;
    }

    function describeSeries(data, column) {
        if (!column) return null;
        const values = data.map((row) => finiteNumber(row, column)).filter((value) => value !== null);
        if (!values.length) return null;
        return {
            count: values.length,
            latest: values[values.length - 1],
            average: values.reduce((sum, value) => sum + value, 0) / values.length,
            min: Math.min(...values),
            max: Math.max(...values)
        };
    }

    function pearsonForRows(data, firstColumn, secondColumn) {
        if (!firstColumn || !secondColumn) return null;
        const pairs = data.map((row) => [finiteNumber(row, firstColumn), finiteNumber(row, secondColumn)])
            .filter((pair) => pair[0] !== null && pair[1] !== null);
        if (pairs.length < 3) return null;
        const avgX = pairs.reduce((sum, pair) => sum + pair[0], 0) / pairs.length;
        const avgY = pairs.reduce((sum, pair) => sum + pair[1], 0) / pairs.length;
        let numerator = 0, xSquare = 0, ySquare = 0;
        pairs.forEach(([x, y]) => {
            const dx = x - avgX, dy = y - avgY;
            numerator += dx * dy; xSquare += dx * dx; ySquare += dy * dy;
        });
        if (!xSquare || !ySquare) return null;
        return { value: numerator / Math.sqrt(xSquare * ySquare), count: pairs.length };
    }

    function getDataAnalysis() {
        const aqua = window.AquaMindData;
        const data = aqua && typeof aqua.getData === 'function' ? aqua.getData() : null;
        const meta = aqua && typeof aqua.getMeta === 'function' ? aqua.getMeta() : null;
        if (!Array.isArray(data) || !data.length) return { hasData: false, count: 0 };

        const headers = Object.keys(data[0] || {});
        const map = typeof aqua.detectColumns === 'function' ? aqua.detectColumns(headers) : {};
        const timeValue = (row) => {
            if (!map.timestamp || !row[map.timestamp]) return null;
            const date = new Date(String(row[map.timestamp]).replace(/-/g, '/'));
            return Number.isNaN(date.getTime()) ? null : date;
        };
        const sorted = data.slice().sort((a, b) => (timeValue(a)?.getTime() || 0) - (timeValue(b)?.getTime() || 0));
        const latestRow = sorted[sorted.length - 1];
        const validTimes = sorted.map(timeValue).filter(Boolean);
        const intervals = [];
        for (let index = 1; index < validTimes.length; index += 1) {
            const hours = (validTimes[index] - validTimes[index - 1]) / 3600000;
            if (hours > 0 && hours <= 24) intervals.push(hours);
        }
        intervals.sort((a, b) => a - b);
        const intervalHours = intervals.length ? intervals[Math.floor(intervals.length / 2)] : null;
        const turbidity = describeSeries(sorted, map.turbidity);
        const effluentTurbidity = describeSeries(sorted, map.effluentTurbidity);
        const ph = describeSeries(sorted, map.ph);
        const chlorine = describeSeries(sorted, map.chlorine);
        const dosage = describeSeries(sorted, map.dosage);
        const power = describeSeries(sorted, map.power);
        const health = describeSeries(sorted, map.health);
        const dosageCorrelation = pearsonForRows(sorted, map.turbidity, map.dosage);
        const exceedCount = map.turbidity ? sorted.filter((row) => {
            const value = finiteNumber(row, map.turbidity);
            return value !== null && value >= 5;
        }).length : null;
        const mappedColumns = Object.values(map).filter(Boolean);
        const validCells = mappedColumns.reduce((sum, column) => sum + sorted.filter((row) => {
            if (column === map.timestamp) return Boolean(timeValue(row));
            return finiteNumber(row, column) !== null;
        }).length, 0);
        const completeness = mappedColumns.length ? validCells / (mappedColumns.length * sorted.length) * 100 : 0;
        let totalEnergy = null;
        if (power) {
            const values = sorted.map((row) => finiteNumber(row, map.power)).filter((value) => value !== null);
            totalEnergy = /kwh|电量/i.test(map.power || '')
                ? values.reduce((sum, value) => sum + value, 0)
                : intervalHours ? values.reduce((sum, value) => sum + value, 0) * intervalHours : null;
        }
        return {
            hasData: true, data: sorted, map, headers, meta, count: sorted.length, latestRow,
            firstTime: validTimes.length ? validTimes[0] : null,
            lastTime: validTimes.length ? validTimes[validTimes.length - 1] : null,
            intervalHours, completeness, turbidity, effluentTurbidity, ph, chlorine, dosage, power, health,
            dosageCorrelation, exceedCount, totalEnergy
        };
    }

    function formatSeries(label, series, unit) {
        if (!series) return `${label}：缺少有效字段或有效数值`;
        return `${label}：最新 ${series.latest.toFixed(2)} ${unit}，平均 ${series.average.toFixed(2)} ${unit}，范围 ${series.min.toFixed(2)}–${series.max.toFixed(2)} ${unit}（有效 ${series.count} 条）`;
    }

    function getAnalysisFacts(analysis) {
        if (!analysis.hasData) return '当前没有已上传的共享数据。';
        const lines = [
            `数据源：${analysis.meta?.fileName || '已上传文件'}`,
            `记录数：${analysis.count} 条；字段：${analysis.headers.join('、')}`,
            `时间范围：${analysis.firstTime ? analysis.firstTime.toLocaleString('zh-CN') : '无法识别'} 至 ${analysis.lastTime ? analysis.lastTime.toLocaleString('zh-CN') : '无法识别'}`,
            `数据完整率：${analysis.completeness.toFixed(1)}%${analysis.intervalHours ? `；典型采样间隔 ${analysis.intervalHours.toFixed(2)} 小时` : ''}`,
            formatSeries('浊度', analysis.turbidity, 'NTU'),
            analysis.turbidity ? `浊度达到或超过 5.0 NTU：${analysis.exceedCount} 条，占有效浊度记录的 ${(analysis.exceedCount / analysis.turbidity.count * 100).toFixed(1)}%` : '浊度超标情况：无法计算',
            formatSeries('出水浊度', analysis.effluentTurbidity, 'NTU'),
            formatSeries('pH', analysis.ph, ''),
            formatSeries('余氯', analysis.chlorine, 'mg/L'),
            formatSeries('加药量', analysis.dosage, 'mg/L'),
            formatSeries('功率/电量字段', analysis.power, /kwh|电量/i.test(analysis.map.power || '') ? 'kWh' : 'kW'),
            analysis.totalEnergy === null ? '当前数据周期能耗：缺少采样间隔或功率/电量字段，无法可靠计算' : `当前数据周期能耗：${analysis.totalEnergy.toFixed(2)} kWh`,
            formatSeries('设备健康度', analysis.health, '%'),
            analysis.dosageCorrelation ? `浊度与加药量 Pearson r=${analysis.dosageCorrelation.value.toFixed(3)}（同一行有效配对 ${analysis.dosageCorrelation.count} 条；相关性不代表因果关系）` : '浊度与加药量相关性：有效配对不足或指标无波动，无法计算'
        ];
        return lines.join('\n');
    }

    function buildLocalDataAnswer(question) {
        const analysis = getDataAnalysis();
        if (!analysis.hasData) return '当前没有可分析的上传数据。请先上传 CSV 文件；上传后我会基于真实字段和有效记录回答，不会使用模拟值。';
        const query = String(question || '').toLowerCase();
        const source = `数据源：${analysis.meta?.fileName || '已上传文件'}，共 ${analysis.count} 条记录。`;
        const quality = `数据完整率 ${analysis.completeness.toFixed(1)}%${analysis.intervalHours ? `，典型采样间隔 ${analysis.intervalHours.toFixed(2)} 小时` : '，采样间隔无法识别'}。`;
        if (/浊度|水质|异常|风险|超标/.test(query)) {
            const details = [formatSeries('浊度', analysis.turbidity, 'NTU')];
            if (analysis.turbidity) details.push(`≥5.0 NTU 的记录 ${analysis.exceedCount} 条，占 ${(analysis.exceedCount / analysis.turbidity.count * 100).toFixed(1)}%。`);
            if (analysis.ph) details.push(formatSeries('pH', analysis.ph, ''));
            if (analysis.chlorine) details.push(formatSeries('余氯', analysis.chlorine, 'mg/L'));
            return `${source}\n${details.join('\n')}\n${quality}以上是数据事实；若缺少工艺段标准，不能仅据此判定全部水质是否合格。`;
        }
        if (/加药|药剂|投加/.test(query)) {
            const correlation = analysis.dosageCorrelation ? `浊度与加药量 Pearson r=${analysis.dosageCorrelation.value.toFixed(3)}，有效配对 ${analysis.dosageCorrelation.count} 条。` : '浊度与加药量的有效配对不足，无法可靠计算相关性。';
            return `${source}\n${formatSeries('加药量', analysis.dosage, 'mg/L')}\n${formatSeries('浊度', analysis.turbidity, 'NTU')}\n${correlation}\n相关性不代表因果关系；缺少流量、药剂浓度和出水目标时，我不会给出具体调药指令。`;
        }
        if (/能耗|功率|电量|电费|节能/.test(query)) {
            const energy = analysis.totalEnergy === null ? '缺少有效功率/电量字段或采样间隔，无法可靠计算周期能耗。' : `当前数据周期能耗为 ${analysis.totalEnergy.toFixed(2)} kWh。`;
            return `${source}\n${formatSeries('功率/电量字段', analysis.power, /kwh|电量/i.test(analysis.map.power || '') ? 'kWh' : 'kW')}\n${energy}\n缺少峰、平、谷电价配置时，无法实事求是计算电费或节省金额。`;
        }
        if (/设备|健康|维护|故障/.test(query)) return `${source}\n${formatSeries('设备健康度', analysis.health, '%')}\n如果文件中没有设备与健康字段的一一映射，就不能判断具体哪台设备需要维护。`;
        if (/质量|缺失|完整|字段|数据源/.test(query)) return `${source}\n${quality}\n识别到的字段：${Object.entries(analysis.map).map(([key, value]) => `${key}=${value}`).join('；') || '无'}。`;
        return `${source}\n${quality}\n${formatSeries('浊度', analysis.turbidity, 'NTU')}\n${formatSeries('加药量', analysis.dosage, 'mg/L')}\n${analysis.totalEnergy === null ? '周期能耗无法可靠计算。' : `当前数据周期能耗 ${analysis.totalEnergy.toFixed(2)} kWh。`}\n${formatSeries('设备健康度', analysis.health, '%')}\n以上仅引用上传数据及确定性计算结果；缺失字段不参与结论。`;
    }

    async function canUseConfiguredModel() {
        try {
            const response = await fetch(`${API_BASE}/api/chat/info`, { cache: 'no-store' });
            if (!response.ok) return true;
            const info = await response.json();
            return info.enabled !== false && info.api_configured !== false;
        } catch (error) {
            return true;
        }
    }

    function getReliableSystemPrompt() {
        return [
            '你是 AquaMind Pro 智慧水务系统的专业数据分析助手。',
            '只能基于下方“已计算事实”和用户提供的信息回答，不得编造实时数据、设备状态、天气、成本或预测结果。',
            '如果回答所需字段缺失，要明确指出缺失字段和无法计算的内容。',
            '区分数据事实、统计关联和操作建议；相关性不代表因果关系。高风险操作必须建议人工确认。',
            `当前页面：${getPageTitle()}`,
            '已计算事实：',
            getAnalysisFacts(getDataAnalysis())
        ].join('\n');
    }

    async function sendMessage(text) {
        const content = String(text || '').trim();
        if (!content || sending) return;

        messages.push({ role: 'user', content });
        saveMessages();
        renderMessages();
        elements.input.value = '';
        setSending(true);

        try {
            if (!(await canUseConfiguredModel())) {
                messages.push({ role: 'assistant', content: buildLocalDataAnswer(content) });
                return;
            }
            const response = await fetch(`${API_BASE}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: [
                        { role: 'system', content: getReliableSystemPrompt() },
                        ...messages.slice(-12).map((item) => ({ role: item.role, content: item.content }))
                    ],
                    temperature: 0.4,
                    max_tokens: 1600
                })
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            const answer = extractAssistantText(data);
            if (!answer) throw new Error(`Empty AI response: ${JSON.stringify(data).slice(0, 300)}`);
            messages.push({ role: 'assistant', content: answer });
        } catch (error) {
            messages.push({
                role: 'assistant',
                content: 'AI 服务暂不可用。请确认后端已启动，并检查 /api/chat 或 LLM 配置。'
            });
        } finally {
            saveMessages();
            renderMessages();
            setSending(false);
        }
    }

    function bindEvents() {
        elements.toggle.addEventListener('click', () => setOpen(!elements.root.classList.contains('is-open')));
        elements.close.addEventListener('click', () => setOpen(false));
        elements.form.addEventListener('submit', (event) => {
            event.preventDefault();
            sendMessage(elements.input.value);
        });
        elements.input.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage(elements.input.value);
            }
        });
        elements.quick.addEventListener('click', (event) => {
            const button = event.target.closest('button[data-question]');
            if (button) sendMessage(button.dataset.question);
        });
    }

    function init() {
        createShell();
        if (!elements.root) return;
        loadMessages();
        renderMessages();
        bindEvents();
    }

    window.AquaMindAssistant = {
        open: function () {
            if (elements.root) setOpen(true);
        },
        ask: sendMessage
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
