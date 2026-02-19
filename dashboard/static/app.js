// Perps Bot Dashboard Logic

const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
let socket;
let scanCount = 0;
let dailyPnl = 0;
let latency = 0;

// Asset switching
let currentAsset = 'ETH';
let latestByAsset = {};  // Store latest scan per asset

function connect() {
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        document.getElementById('ws-badge').textContent = '● Connected';
        document.getElementById('ws-badge').classList.add('connected');
        console.log('WS Connected');
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };

    socket.onclose = () => {
        document.getElementById('ws-badge').textContent = '● Disconnected';
        document.getElementById('ws-badge').classList.remove('connected');
        setTimeout(connect, 3000);
    };
}

function handleMessage(data) {
    if (data.type === 'init') {
        renderHistory(data.history);
        updateBufferInfo(data.buffer_info);
        if (data.latest) {
            const asset = data.latest.symbol || 'ETH';
            latestByAsset[asset] = data.latest;
            if (asset === currentAsset) updateLatest(data.latest);
        }
        scanCount = data.total_scans;
        updateStats();
    } else if (data.type === 'pong') {
        updateBufferInfo(data.buffer_info);
    } else if (data.type === 'price_tick') {
        // Live price tick — update if matches selected asset
        const asset = data.asset || 'ETH';
        if (asset === currentAsset) {
            document.getElementById('mark-price').textContent = `$${data.price.toFixed(2)}`;
        }
    } else {
        // Real-time scan result
        scanCount++;
        renderRow(data);
        const asset = data.symbol || 'ETH';
        latestByAsset[asset] = data;
        if (asset === currentAsset) {
            updateLatest(data);
            // Update price from scan
            const price = data.market_price || data.mark_price;
            if (price) document.getElementById('mark-price').textContent = `$${price.toFixed(2)}`;
        }
        updateStats();
    }
}

function updateBufferInfo(info) {
    if (!info) return;

    // Per-asset price and funding
    const assetKey = currentAsset.toLowerCase();
    const markPrice = info.mark_price || info[`${assetKey}_mark`] || 0;
    const funding = info[`${assetKey}_funding`] || info.funding || 0;
    const trades = info[`${assetKey}_trades`] || 0;

    document.getElementById('mark-price').textContent =
        markPrice ? `$${markPrice.toFixed(2)}` : '--';

    const fundEl = document.getElementById('funding-rate');
    const fundRate = parseFloat(funding);
    fundEl.textContent = `${(fundRate * 100).toFixed(4)}%`;
    fundEl.style.color = fundRate > 0.01 ? '#ff4d4d' : (fundRate < -0.01 ? '#00e676' : 'var(--text-secondary)');

    const tradeEl = document.getElementById('trade-count-display');
    if (tradeEl) tradeEl.textContent = trades;

    // Update Data Feeds list
    if (document.getElementById('feed-count-eth')) {
        document.getElementById('feed-count-eth').textContent = info.eth_trades || 0;
        document.getElementById('feed-count-btc').textContent = info.btc_trades || 0;
        document.getElementById('feed-count-sol').textContent = info.sol_trades || 0;

        document.getElementById('feed-count-eth-bn').textContent = info.eth_binance_trades || 0;
        document.getElementById('feed-count-btc-bn').textContent = info.btc_binance_trades || 0;
        document.getElementById('feed-count-sol-bn').textContent = info.sol_binance_trades || 0;
    }
}

function updateLatest(scan) {
    if (!scan) return;

    // Update Signal Gauges
    updateGauge('mtf', scan.mtf_score, scan.mtf_bias); // mtf_bias is 'bullish'/'bearish'
    updateGauge('ofi', scan.ofi_score);
    updateGauge('corr', scan.corr_score);
    updateGauge('fund', scan.fund_score);
    updateGauge('vp', scan.vp_score);

    // Update PnL & Score
    if (scan.daily_pnl !== undefined) dailyPnl = scan.daily_pnl;
    if (scan.score !== undefined) document.getElementById('last-score').textContent = scan.score;

    // Latency
    const now = Date.now();
    // approximate latency from server timestamp if available, else just placeholder
    // scan.timestamp is seconds
    if (scan.timestamp) {
        latency = Math.floor((now / 1000 - scan.timestamp) * 1000);
        if (latency < 0) latency = 0; // clock skew
        document.getElementById('latency').textContent = `${latency}ms`;
    }
}

function updateStats() {
    document.getElementById('scan-count').textContent = scanCount;
    const pnlEl = document.getElementById('daily-pnl');
    pnlEl.textContent = `$${dailyPnl.toFixed(2)}`;
    pnlEl.style.color = dailyPnl >= 0 ? '#00e676' : '#ff4d4d';
}

function renderHistory(history) {
    const tbody = document.getElementById('log-tbody');
    tbody.innerHTML = '';
    // Show last 20
    history.slice(-20).reverse().forEach(row => {
        const tr = createRow(row);
        tbody.appendChild(tr);
    });
}

function renderRow(data) {
    const tbody = document.getElementById('log-tbody');
    const tr = createRow(data);
    tr.classList.add('new-row');
    tbody.prepend(tr);
    if (tbody.children.length > 20) {
        tbody.lastChild.remove();
    }
}

function createRow(data) {
    const tr = document.createElement('tr');

    // Time
    const timeTd = document.createElement('td');
    timeTd.textContent = data.time_str || new Date().toLocaleTimeString();

    // Asset
    const assetTd = document.createElement('td');
    assetTd.textContent = data.asset || data.symbol || 'ETH';
    assetTd.style.color = '#8b8fa3';

    // Action
    const actionTd = document.createElement('td');
    const action = data.action || 'WAIT';
    actionTd.textContent = action;
    actionTd.className = getActionClass(action);

    // MTF
    const mtfTd = document.createElement('td');
    mtfTd.textContent = data.mtf_bias || 'NONE';
    mtfTd.className = data.mtf_bias === 'BULL' ? 'td-green' : (data.mtf_bias === 'BEAR' ? 'td-red' : '');

    // OFI
    const ofiTd = document.createElement('td');
    const ofi = data.ofi_score !== undefined ? data.ofi_score : '-';
    ofiTd.textContent = ofi;
    if (typeof ofi === 'number') {
        ofiTd.className = ofi > 0 ? 'td-green' : (ofi < 0 ? 'td-red' : '');
    }

    // LVN / VP
    const lvnTd = document.createElement('td');
    // data.vp_score is 1 if in_lvn, else 0
    lvnTd.textContent = data.vp_score ? '✓' : '-';
    if (data.vp_score) lvnTd.className = 'td-green';

    // Price
    const priceTd = document.createElement('td');
    const p = data.mark_price || data.market_price;
    priceTd.textContent = p ? `$${p.toFixed(2)}` : '-';

    // Score
    const scoreTd = document.createElement('td');
    scoreTd.textContent = data.score || '0';

    tr.append(timeTd, assetTd, actionTd, mtfTd, ofiTd, lvnTd, priceTd, scoreTd);
    return tr;
}

function getActionClass(action) {
    if (action.includes('LONG')) return 'text-success';
    if (action.includes('SHORT')) return 'text-danger';
    return 'text-muted';
}

function updateGauge(id, value, bias) {
    const valEl = document.getElementById(`gauge-val-${id}`);
    const fillEl = document.getElementById(`gauge-fill-${id}`);
    if (!valEl) return;

    let displayVal = value !== undefined ? value : '-';
    valEl.textContent = displayVal;

    // Simple fill logic (0 to 1 range assumed for some, -1 to 1 for others)
    // Customize based on signal types if needed.
    // For now just color based on positive/negative
    if (typeof value === 'number') {
        if (value > 0) fillEl.style.stroke = '#00e676';
        else if (value < 0) fillEl.style.stroke = '#ff4d4d';
        else fillEl.style.stroke = '#888';

        // Stroke dasharray for circle (circumference ~251)
        // map -1..1 to 0..251 ? or 0..1 to 0..251?
        // Let's just do full circle for active, gray for inactive
        fillEl.style.strokeDasharray = '251';
        fillEl.style.strokeDashoffset = '0';
    }
}

function switchAsset(asset) {
    currentAsset = asset;
    document.querySelectorAll('.asset-tab').forEach(b => b.classList.remove('active-tab'));
    document.querySelector(`button[data-asset="${asset}"]`).classList.add('active-tab');
    // Re-render with this asset's latest data
    if (latestByAsset[asset]) {
        updateLatest(latestByAsset[asset]);
        const price = latestByAsset[asset].market_price || latestByAsset[asset].mark_price;
        if (price) document.getElementById('mark-price').textContent = `$${price.toFixed(2)}`;
    }
    // Re-request buffer info for this asset
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send('ping');
    }
}

// Start
connect();

// Time updater
setInterval(() => {
    document.getElementById('time-badge').textContent = new Date().toLocaleTimeString();
}, 1000);

// Ping for buffer updates
setInterval(() => {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send('ping');
    }
}, 2000);
