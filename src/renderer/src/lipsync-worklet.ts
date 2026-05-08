class OttTuberLipSyncCaptureProcessor extends AudioWorkletProcessor {
	process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
		const inputChannels = inputs[0];
		const outputChannels = outputs[0];
		if (inputChannels && inputChannels.length > 0) {
			const frameLength = inputChannels[0].length;
			const mono = new Float32Array(frameLength);
			for (let channel = 0; channel < inputChannels.length; channel++) {
				const samples = inputChannels[channel];
				for (let i = 0; i < frameLength; i++) {
					mono[i] += samples[i] / inputChannels.length;
				}
			}
			this.port.postMessage(mono, [mono.buffer]);
		}

		if (outputChannels && outputChannels.length > 0) {
			for (const channel of outputChannels) {
				channel.fill(0);
			}
		}

		return true;
	}
}

registerProcessor("otttuber-lipsync-capture", OttTuberLipSyncCaptureProcessor);
