let audioCtx = null;

function initAudio() {
  if (!window.AudioContext && !window.webkitAudioContext) return;
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
}

// Highly optimized, zero-latency procedural synth tick (Safe transaction)
export const playTick = () => {
  try {
    initAudio();
    if (!audioCtx) return;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    
    // Quick, futuristic high-pitch blip
    osc.type = "sine";
    osc.frequency.setValueAtTime(1000, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(400, audioCtx.currentTime + 0.05);
    
    gain.gain.setValueAtTime(0.15, audioCtx.currentTime); // keep volume subtle
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.05);
    
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    
    osc.start();
    osc.stop(audioCtx.currentTime + 0.05);
  } catch (e) {
    // Fail silently so UI never lags
  }
};

// Procedural, authoritative low thud/buzz (Fraud detected)
export const playAlert = () => {
  try {
    initAudio();
    if (!audioCtx) return;
    
    // Dual oscillators for a richer, more serious tone
    const osc1 = audioCtx.createOscillator();
    const osc2 = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    
    osc1.type = "sawtooth";
    osc1.frequency.setValueAtTime(150, audioCtx.currentTime);
    osc1.frequency.exponentialRampToValueAtTime(80, audioCtx.currentTime + 0.2);

    osc2.type = "square";
    osc2.frequency.setValueAtTime(145, audioCtx.currentTime); // slight detune causes dissonance
    osc2.frequency.exponentialRampToValueAtTime(78, audioCtx.currentTime + 0.2);
    
    gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
    gain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.2);
    
    osc1.connect(gain);
    osc2.connect(gain);
    gain.connect(audioCtx.destination);
    
    osc1.start();
    osc2.start();
    osc1.stop(audioCtx.currentTime + 0.2);
    osc2.stop(audioCtx.currentTime + 0.2);
  } catch (e) {}
};
