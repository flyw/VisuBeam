const log = document.getElementById('log');
const wsUrl = `ws://${location.hostname}:${location.port}`;
const httpUrl = `http://${location.hostname}:${location.port}`;

let updateList = [];
let anglesSocket = null;
let audioSockets = {};
const audioContexts = {}; // For real-time playback

function logMessage(message) {
    console.log(message);
    log.textContent = `[${new Date().toLocaleTimeString()}] ${message}
${log.textContent}`;
}

// --- 1. System Info ---
const getDeviceIdBtn = document.getElementById('getDeviceIdBtn');
const postDeviceIdInput = document.getElementById('postDeviceIdInput');
const postDeviceIdBtn = document.getElementById('postDeviceIdBtn');
const deviceIdResultEl = document.getElementById('deviceIdResult');

if (getDeviceIdBtn) {
    getDeviceIdBtn.addEventListener('click', async () => {
        deviceIdResultEl.textContent = 'Fetching...';
        try {
            logMessage('Fetching system device ID...');
            const response = await fetch(`${httpUrl}/api/v1/system/device_id`);
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || 'Request failed');
            logMessage(`Device ID received: ${JSON.stringify(result)}`);
            deviceIdResultEl.textContent = JSON.stringify(result, null, 2);
        } catch (error) {
            logMessage(`Device ID error: ${error.message}`);
            deviceIdResultEl.textContent = `Error: ${error.message}`;
        }
    });
}

if (postDeviceIdBtn) {
    postDeviceIdBtn.addEventListener('click', async () => {
        const deviceId = postDeviceIdInput.value;
        if (!deviceId) {
            alert('Please enter a Device ID.');
            return;
        }
        deviceIdResultEl.textContent = 'Posting...';
        try {
            logMessage(`Posting new device ID: ${deviceId}`);
            const response = await fetch(`${httpUrl}/api/v1/system/device_id`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ device_id: deviceId }),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || 'Request failed');
            logMessage(`Device ID post success: ${JSON.stringify(result)}`);
            deviceIdResultEl.textContent = JSON.stringify(result, null, 2);
        } catch (error) {
            logMessage(`Device ID post error: ${error.message}`);
            deviceIdResultEl.textContent = `Error: ${error.message}`;
        }
    });
}

// --- 2. Update Tracking ---
const updateForm = document.getElementById('update-form');
const updateListEl = document.getElementById('update-list');
const sendUpdateBtn = document.getElementById('send-update');

if (updateForm) {
    updateForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const id = parseInt(document.getElementById('update-id').value);
        const angle = parseFloat(document.getElementById('update-angle').value);
        if (!isNaN(id) && !isNaN(angle)) {
            updateList.push({ id, angle });
            updateListEl.textContent = JSON.stringify(updateList, null, 2);
            updateForm.reset();
        }
    });
}

if (sendUpdateBtn) {
    sendUpdateBtn.addEventListener('click', async () => {
        if (updateList.length === 0) {
            logMessage('Update list is empty.');
            return;
        }
        try {
            logMessage(`Sending update: ${JSON.stringify(updateList)}`);
            const response = await fetch(`${httpUrl}/api/v1/tracking/update`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updateList),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || 'Request failed');
            logMessage(`Update success: ${JSON.stringify(result)}`);
            updateList = [];
            updateListEl.textContent = '[]';
        } catch (error) {
            logMessage(`Update error: ${error.message}`);
        }
    });
}

// --- 3. Person Leave ---
const leaveForm = document.getElementById('leave-form');
if (leaveForm) {
    leaveForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const id = parseInt(document.getElementById('leave-id').value);
        if (isNaN(id)) return;
        try {
            logMessage(`Sending leave for ID: ${id}`);
            const response = await fetch(`${httpUrl}/api/v1/tracking/leave`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id }),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || 'Request failed');
            logMessage(`Leave success: ${JSON.stringify(result)}`);
        } catch (error) {
            logMessage(`Leave error: ${error.message}`);
        }
    });
}

// --- 4. Angle Stream ---
const connectAnglesBtn = document.getElementById('connect-angles');
const disconnectAnglesBtn = document.getElementById('disconnect-angles');
const anglesDataEl = document.getElementById('angles-data');

if (connectAnglesBtn) {
    connectAnglesBtn.addEventListener('click', () => {
        if (anglesSocket) {
            logMessage('Angle stream already connected.');
            return;
        }
        anglesSocket = new WebSocket(`${wsUrl}/ws/tracking/angles`);
        
        anglesSocket.onopen = () => {
            logMessage('Angle stream connected.');
            anglesDataEl.textContent = 'Connected. Waiting for data...';
            connectAnglesBtn.disabled = true;
            disconnectAnglesBtn.disabled = false;
        };
        
        anglesSocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            logMessage(`Received angles: ${event.data}`);
            anglesDataEl.textContent = JSON.stringify(data, null, 2);
        };

        anglesSocket.onclose = (event) => {
            logMessage(`Angle stream disconnected. Code: ${event.code}, Reason: ${event.reason}`);
            anglesDataEl.textContent = 'Not connected';
            anglesSocket = null;
            connectAnglesBtn.disabled = false;
            disconnectAnglesBtn.disabled = true;
        };

        anglesSocket.onerror = (error) => {
            logMessage(`Angle stream error: ${error.message || 'An unknown error occurred.'}`);
        };
    });
}

if (disconnectAnglesBtn) {
    disconnectAnglesBtn.addEventListener('click', () => {
        if (anglesSocket) {
            anglesSocket.close();
        }
    });
}

// --- 5. Enhanced Audio (Real-time Playback) ---
const audioForm = document.getElementById('audio-form');
const audioStreamsContainer = document.getElementById('audio-streams');

if (audioForm) {
    audioForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const personId = document.getElementById('audio-id').value;
        if (!personId) return;

        if (audioSockets[personId]) {
            logMessage(`Audio stream for person ID ${personId} is already connected.`);
            return;
        }

        const newSocket = new WebSocket(`${wsUrl}/ws/audio/enhanced/${personId}`);
        newSocket.binaryType = 'arraybuffer'; // Important for receiving raw binary data
        logMessage(`Attempting to connect audio stream for person ID: ${personId}`);
        
        const streamDiv = document.createElement('div');
        streamDiv.id = `audio-stream-${personId}`;
        streamDiv.innerHTML = `
            <h4>Stream for Person ID: ${personId}</h4>
            <pre>Connecting...</pre>
            <button class="disconnect">Disconnect</button>
        `;
        audioStreamsContainer.appendChild(streamDiv);

        newSocket.onopen = () => {
            logMessage(`Audio stream connected for person ID: ${personId}`);
            streamDiv.querySelector('pre').textContent = 'Connected. Waiting for data...';
            audioSockets[personId] = newSocket;
            
            // Setup Web Audio API for this stream
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            audioContexts[personId] = {
                context: audioCtx,
                nextPlayTime: audioCtx.currentTime,
                personId: personId
            };
        };

        newSocket.onmessage = async (event) => {
            const audioState = audioContexts[personId];
            if (!audioState || !(event.data instanceof ArrayBuffer)) return;

            const pre = streamDiv.querySelector('pre');
            const receiveTime = new Date().toLocaleTimeString();
            pre.textContent = `[${receiveTime}] Playing chunk. Size: ${event.data.byteLength} bytes.`;

            const pcmData = new Int16Array(event.data);
            const float32Data = new Float32Array(pcmData.length);
            for (let i = 0; i < pcmData.length; i++) {
                float32Data[i] = pcmData[i] / 32768.0; // Convert 16-bit PCM to [-1.0, 1.0] float
            }

            // THE FIX: Create buffer with the correct 16kHz sample rate
            const audioBuffer = audioState.context.createBuffer(1, float32Data.length, 16000);
            audioBuffer.getChannelData(0).set(float32Data);
            
            const source = audioState.context.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioState.context.destination);

            // Schedule playback seamlessly
            const currentTime = audioState.context.currentTime;
            if (audioState.nextPlayTime < currentTime) {
                audioState.nextPlayTime = currentTime;
            }
            source.start(audioState.nextPlayTime);
            audioState.nextPlayTime += audioBuffer.duration;
        };

        const cleanup = () => {
            logMessage(`Cleaning up audio resources for person ID: ${personId}`);
            delete audioSockets[personId];
            const audioState = audioContexts[personId];
            if (audioState) {
                audioState.context.close();
                delete audioContexts[personId];
            }
            streamDiv.remove();
        };

        newSocket.onclose = (event) => {
            logMessage(`Audio stream for ${personId} disconnected. Reason: ${event.reason}`);
            cleanup();
        };

        newSocket.onerror = (error) => {
            logMessage(`Audio stream error for ${personId}: ${error.message || 'An unknown error occurred.'}`);
            cleanup();
        };

        streamDiv.querySelector('button.disconnect').addEventListener('click', () => {
            if (audioSockets[personId]) {
                audioSockets[personId].close();
            }
        });
    });
}
