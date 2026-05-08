import type { VRM } from "@pixiv/three-vrm";
import * as ort from "onnxruntime-web/all";

type DeviceConfig = {
	deviceLabel?: string | null;
	deviceId?: string | null;
};

type OpenLipSyncConfig = {
	model?: {
		num_visemes?: number;
	};
	audio?: {
		sample_rate: number;
		hop_length_ms?: number;
		window_length_ms?: number;
		n_fft: number;
		n_mels: number;
		fmin: number;
		fmax: number;
	};
	training?: {
		multi_label?: boolean;
	};
};

type AudioProcessingConfig = {
	sampleRate: number;
	hopLengthSamples: number;
	windowLengthSamples: number;
	nFft: number;
	nMels: number;
	fMin: number;
	fMax: number;
};

export const OPEN_LIPSYNC_VISEMES = [
	"sil",
	"PP",
	"FF",
	"TH",
	"DD",
	"kk",
	"CH",
	"SS",
	"nn",
	"RR",
	"aa",
	"E",
	"ih",
	"oh",
	"ou",
] as const;

export const STANDARD_MOUTH_EXPRESSIONS = [
	"aa",
	"ee",
	"ih",
	"oh",
	"ou",
] as const;

export const WEBCAM_MOUTH_SOURCE_EXPRESSIONS = [
	"jawOpen",
	"mouthClose",
	"mouthPucker",
	"mouthStretchLeft",
	"mouthStretchRight",
	"mouthFunnel",
] as const;

type OpenLipSyncVisemeName = (typeof OPEN_LIPSYNC_VISEMES)[number];
export type StandardMouthExpression =
	(typeof STANDARD_MOUTH_EXPRESSIONS)[number];
type ProviderName = "webnn" | "webgpu" | "webgl" | "wasm";
type WeightMap<T extends string> = Record<T, number>;
type ExpressionBinding<T extends string> = Partial<Record<T, string>>;

type LipSyncState = {
	enabled: boolean;
	active: boolean;
	provider: ProviderName | null;
	customVisemes: string[];
	standardMouths: string[];
	visemes: WeightMap<OpenLipSyncVisemeName>;
	mouth: WeightMap<StandardMouthExpression>;
};

export type VrmExpressionRouting = {
	customVisemes: ExpressionBinding<OpenLipSyncVisemeName>;
	standardMouths: ExpressionBinding<StandardMouthExpression>;
	mouthSourceExpressions: string[];
	customVisemeNames: string[];
	standardMouthNames: string[];
};

const PROVIDER_ORDER: ProviderName[] = ["webnn", "webgpu", "webgl", "wasm"];

const OPEN_LIPSYNC_TO_CUSTOM_VISEME: Partial<
	Record<OpenLipSyncVisemeName, string>
> = {
	sil: "SIL",
	PP: "PP",
	FF: "FF",
	TH: "TH",
	DD: "DD",
	kk: "KK",
	CH: "CH",
	SS: "SS",
	nn: "NN",
	RR: "RR",
};

const VISEME_TO_STANDARD_MOUTH: Record<
	StandardMouthExpression,
	OpenLipSyncVisemeName
> = {
	aa: "aa",
	ee: "E",
	ih: "ih",
	oh: "oh",
	ou: "ou",
};

function zeroVisemeWeights(): WeightMap<OpenLipSyncVisemeName> {
	return Object.fromEntries(
		OPEN_LIPSYNC_VISEMES.map((name) => [name, name === "sil" ? 1 : 0]),
	) as WeightMap<OpenLipSyncVisemeName>;
}

function zeroMouthWeights(): WeightMap<StandardMouthExpression> {
	return Object.fromEntries(
		STANDARD_MOUTH_EXPRESSIONS.map((name) => [name, 0]),
	) as WeightMap<StandardMouthExpression>;
}

function clamp01(value: number): number {
	return Math.max(0, Math.min(1, value));
}

function resolveSiblingConfigPath(modelPath: string): string {
	const normalized = modelPath.replaceAll("\\", "/");
	const slashIndex = normalized.lastIndexOf("/");
	return slashIndex === -1 ? "config.json" : (
			`${normalized.slice(0, slashIndex)}/config.json`
		);
}

function parseAudioProcessingConfig(
	config: OpenLipSyncConfig,
): AudioProcessingConfig {
	const audio = config.audio;
	if (!audio) {
		throw new Error("Lip-sync model config is missing its audio section");
	}

	const sampleRate = audio.sample_rate;
	const hopLengthSamples =
		audio.hop_length_ms && audio.hop_length_ms > 0 ?
			Math.round((sampleRate * audio.hop_length_ms) / 1000)
		:	Math.round(sampleRate / 100);
	const windowLengthSamples =
		audio.window_length_ms && audio.window_length_ms > 0 ?
			Math.round((sampleRate * audio.window_length_ms) / 1000)
		:	Math.round(sampleRate * 0.025);

	return {
		sampleRate,
		hopLengthSamples,
		windowLengthSamples,
		nFft: audio.n_fft,
		nMels: audio.n_mels,
		fMin: audio.fmin,
		fMax: audio.fmax,
	};
}

function createMediaConstraint(
	kind: MediaDeviceKind,
	device?: DeviceConfig | null,
): MediaTrackConstraints | boolean {
	if (!device) {
		return true;
	}
	if (device.deviceLabel) {
		return navigator.mediaDevices.enumerateDevices().then((devices) => {
			const match = devices.find(
				(candidate) =>
					candidate.kind === kind &&
					candidate.label === device.deviceLabel,
			);
			return (
				match ? { deviceId: { exact: match.deviceId } }
				: device.deviceId ? { deviceId: { exact: device.deviceId } }
				: true
			);
		});
	}
	if (device.deviceId) {
		return { deviceId: { exact: device.deviceId } };
	}
	return true;
}

async function openMicrophone(
	device?: DeviceConfig | null,
): Promise<MediaStream> {
	const audioConstraint = await createMediaConstraint("audioinput", device);
	return navigator.mediaDevices.getUserMedia({
		audio: {
			...(audioConstraint === true ? {} : audioConstraint),
			channelCount: 1,
			echoCancellation: false,
			noiseSuppression: false,
			autoGainControl: false,
		},
	});
}

function hzToMel(hz: number): number {
	return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number): number {
	return 700 * (10 ** (mel / 2595) - 1);
}

class FftProcessor {
	private readonly size: number;
	private readonly real: Float32Array;
	private readonly imag: Float32Array;
	private readonly bitReversal: Uint32Array;

	constructor(size: number) {
		if (size <= 0 || (size & (size - 1)) !== 0) {
			throw new Error("FFT size must be a positive power of two");
		}
		this.size = size;
		this.real = new Float32Array(size);
		this.imag = new Float32Array(size);
		this.bitReversal = new Uint32Array(size);
		const bits = Math.log2(size);
		for (let i = 0; i < size; i++) {
			let reversed = 0;
			let value = i;
			for (let bit = 0; bit < bits; bit++) {
				reversed = (reversed << 1) | (value & 1);
				value >>= 1;
			}
			this.bitReversal[i] = reversed;
		}
	}

	forward(input: Float32Array, output: Float32Array): void {
		for (let i = 0; i < this.size; i++) {
			const reversed = this.bitReversal[i];
			this.real[i] = reversed < input.length ? input[reversed] : 0;
			this.imag[i] = 0;
		}

		for (let len = 2; len <= this.size; len <<= 1) {
			const angle = (-2 * Math.PI) / len;
			const cos = Math.cos(angle);
			const sin = Math.sin(angle);
			for (let start = 0; start < this.size; start += len) {
				let wr = 1;
				let wi = 0;
				for (let offset = 0; offset < len / 2; offset++) {
					const even = start + offset;
					const odd = even + len / 2;
					const oddReal = this.real[odd] * wr - this.imag[odd] * wi;
					const oddImag = this.real[odd] * wi + this.imag[odd] * wr;
					const evenReal = this.real[even];
					const evenImag = this.imag[even];

					this.real[even] = evenReal + oddReal;
					this.imag[even] = evenImag + oddImag;
					this.real[odd] = evenReal - oddReal;
					this.imag[odd] = evenImag - oddImag;

					const nextWr = wr * cos - wi * sin;
					wi = wr * sin + wi * cos;
					wr = nextWr;
				}
			}
		}

		const outputLength = output.length;
		for (let i = 0; i < outputLength; i++) {
			output[i] =
				this.real[i] * this.real[i] + this.imag[i] * this.imag[i];
		}
	}
}

class AudioRingBuffer {
	private readonly buffer: Float32Array;
	private readIndex = 0;
	private writeIndex = 0;
	private availableSamples = 0;

	constructor(capacity: number) {
		this.buffer = new Float32Array(capacity);
	}

	get available(): number {
		return this.availableSamples;
	}

	write(samples: Float32Array): void {
		for (let i = 0; i < samples.length; i++) {
			this.buffer[this.writeIndex] = samples[i];
			this.writeIndex = (this.writeIndex + 1) % this.buffer.length;
			if (this.availableSamples < this.buffer.length) {
				this.availableSamples++;
			} else {
				this.readIndex = (this.readIndex + 1) % this.buffer.length;
			}
		}
	}

	readInto(target: Float32Array): boolean {
		if (this.availableSamples < target.length) {
			return false;
		}
		for (let i = 0; i < target.length; i++) {
			target[i] = this.buffer[this.readIndex];
			this.readIndex = (this.readIndex + 1) % this.buffer.length;
		}
		this.availableSamples -= target.length;
		return true;
	}
}

class AudioResampler {
	private readonly inputPerOutput: number;
	private readonly filterTaps: number;
	private readonly halfTaps: number;
	private readonly numPhases: number;
	private readonly coeffTable: Float32Array;
	private readonly buffer: number[] = [];
	private time = 0;
	private primed = false;

	constructor(
		private readonly inputSampleRate: number,
		private readonly outputSampleRate: number,
		filterTaps = 48,
		numPhases = 1024,
		cutoffScale = 0.9,
	) {
		this.inputPerOutput = inputSampleRate / outputSampleRate;
		this.filterTaps = filterTaps;
		this.halfTaps = filterTaps / 2;
		this.numPhases = numPhases;
		this.coeffTable = new Float32Array(numPhases * filterTaps);
		const cutoff =
			0.5 * Math.min(1, outputSampleRate / inputSampleRate) * cutoffScale;
		this.buildCoefficientTable(cutoff);
	}

	resample(input: Float32Array): Float32Array {
		if (input.length > 0) {
			if (!this.primed) {
				for (let i = 0; i < this.halfTaps; i++) {
					this.buffer.push(0);
				}
				this.time = this.halfTaps - 1;
				this.primed = true;
			}
			for (let i = 0; i < input.length; i++) {
				this.buffer.push(input[i]);
			}
		}

		if (this.buffer.length < this.filterTaps) {
			return new Float32Array(0);
		}

		const output: number[] = [];
		for (;;) {
			const center = Math.floor(this.time);
			const leftIndex = center - (this.halfTaps - 1);
			const rightIndex = center + this.halfTaps;
			if (leftIndex < 0 || rightIndex >= this.buffer.length) {
				break;
			}

			const frac = this.time - center;
			let phaseIndex = Math.round(frac * this.numPhases);
			if (phaseIndex === this.numPhases) {
				phaseIndex = 0;
			}
			const coeffBase = phaseIndex * this.filterTaps;
			let sum = 0;
			for (let tap = 0; tap < this.filterTaps; tap++) {
				sum +=
					this.buffer[leftIndex + tap] *
					this.coeffTable[coeffBase + tap];
			}
			output.push(sum);
			this.time += this.inputPerOutput;
		}

		const safeToRemove = Math.floor(this.time) - (this.halfTaps - 1);
		if (safeToRemove > 0) {
			this.buffer.splice(0, Math.min(safeToRemove, this.buffer.length));
			this.time -= safeToRemove;
			if (this.time < 0) {
				this.time = 0;
			}
		}

		return Float32Array.from(output);
	}

	private buildCoefficientTable(cutoff: number): void {
		for (let phase = 0; phase < this.numPhases; phase++) {
			const frac = phase / this.numPhases;
			let sum = 0;
			for (let tap = 0; tap < this.filterTaps; tap++) {
				const t = tap - (this.halfTaps - 1) - frac;
				const sincArg = 2 * cutoff * t;
				const sinc =
					sincArg === 0 ? 1 : (
						Math.sin(Math.PI * sincArg) / (Math.PI * sincArg)
					);
				const window =
					0.42 -
					0.5 *
						Math.cos((2 * Math.PI * tap) / (this.filterTaps - 1)) +
					0.08 *
						Math.cos((4 * Math.PI * tap) / (this.filterTaps - 1));
				const value = 2 * cutoff * sinc * window;
				this.coeffTable[phase * this.filterTaps + tap] = value;
				sum += value;
			}
			if (sum !== 0) {
				const scale = 1 / sum;
				for (let tap = 0; tap < this.filterTaps; tap++) {
					this.coeffTable[phase * this.filterTaps + tap] *= scale;
				}
			}
		}
	}
}

class MelSpectrogramProcessor {
	private readonly window: Float32Array;
	private readonly previousSamples: Float32Array;
	private readonly frameBuffer: Float32Array;
	private readonly fftBuffer: Float32Array;
	private readonly powerSpectrum: Float32Array;
	private readonly melSpectrum: Float32Array;
	private readonly melFilterBank: Float32Array[];
	private readonly fft: FftProcessor;

	constructor(private readonly config: AudioProcessingConfig) {
		this.window = new Float32Array(config.windowLengthSamples);
		for (let i = 0; i < this.window.length; i++) {
			this.window[i] =
				0.5 *
				(1 - Math.cos((2 * Math.PI * i) / (this.window.length - 1)));
		}
		this.previousSamples = new Float32Array(
			config.windowLengthSamples - config.hopLengthSamples,
		);
		this.frameBuffer = new Float32Array(config.windowLengthSamples);
		this.fftBuffer = new Float32Array(config.nFft);
		this.powerSpectrum = new Float32Array(config.nFft / 2 + 1);
		this.melSpectrum = new Float32Array(config.nMels);
		this.melFilterBank = createMelFilterBank(config);
		this.fft = new FftProcessor(config.nFft);
	}

	tryProcessNextHop(ringBuffer: AudioRingBuffer): Float32Array | null {
		const hop = new Float32Array(this.config.hopLengthSamples);
		if (!ringBuffer.readInto(hop)) {
			return null;
		}

		this.frameBuffer.set(this.previousSamples, 0);
		this.frameBuffer.set(hop, this.previousSamples.length);
		this.previousSamples.set(
			this.frameBuffer.subarray(
				this.config.hopLengthSamples,
				this.config.windowLengthSamples,
			),
			0,
		);

		this.fftBuffer.fill(0);
		for (let i = 0; i < this.config.windowLengthSamples; i++) {
			this.fftBuffer[i] = this.frameBuffer[i] * this.window[i];
		}

		this.fft.forward(this.fftBuffer, this.powerSpectrum);

		for (let mel = 0; mel < this.config.nMels; mel++) {
			let sum = 0;
			const filter = this.melFilterBank[mel];
			for (let bin = 0; bin < filter.length; bin++) {
				sum += this.powerSpectrum[bin] * filter[bin];
			}
			this.melSpectrum[mel] = 10 * Math.log10(Math.max(sum, 1e-10));
		}

		return this.melSpectrum.slice();
	}
}

function createMelFilterBank(config: AudioProcessingConfig): Float32Array[] {
	const melMin = hzToMel(config.fMin);
	const melMax = hzToMel(config.fMax);
	const melPoints = new Float32Array(config.nMels + 2);
	for (let i = 0; i < melPoints.length; i++) {
		melPoints[i] = melMin + ((melMax - melMin) * i) / (config.nMels + 1);
	}

	const hzPoints = new Float32Array(melPoints.length);
	for (let i = 0; i < hzPoints.length; i++) {
		hzPoints[i] = melToHz(melPoints[i]);
	}

	const binPoints = new Float32Array(hzPoints.length);
	for (let i = 0; i < hzPoints.length; i++) {
		binPoints[i] = ((config.nFft + 1) * hzPoints[i]) / config.sampleRate;
	}

	return Array.from({ length: config.nMels }, (_, mel) => {
		const filter = new Float32Array(config.nFft / 2 + 1);
		const left = binPoints[mel];
		const center = binPoints[mel + 1];
		const right = binPoints[mel + 2];
		for (let bin = 0; bin < filter.length; bin++) {
			if (bin >= left && bin <= center) {
				filter[bin] = (bin - left) / (center - left);
			} else if (bin > center && bin <= right) {
				filter[bin] = (right - bin) / (right - center);
			}
		}
		return filter;
	});
}

async function createOrtSession(
	model: ArrayBuffer,
): Promise<{ session: ort.InferenceSession; provider: ProviderName }> {
	const providerErrors: string[] = [];
	for (const provider of PROVIDER_ORDER) {
		try {
			const session = await ort.InferenceSession.create(model, {
				executionProviders: [provider],
			});
			return { session, provider };
		} catch (error) {
			const message =
				error instanceof Error ? error.message : String(error);
			providerErrors.push(`[${provider}] ${message}`);
		}
	}
	throw new Error(
		`No ONNX Runtime provider could create a session. ${providerErrors.join(
			", ",
		)}`,
	);
}

export function inspectVrmExpressionRouting(vrm: VRM): VrmExpressionRouting {
	const expressions =
		vrm.expressionManager?.expressions.map(
			(expression) => expression.expressionName,
		) ?? [];
	const byLowercase = new Map(
		expressions.map((expression) => [expression.toLowerCase(), expression]),
	);

	const customVisemes: ExpressionBinding<OpenLipSyncVisemeName> = {};
	for (const [viseme, expressionName] of Object.entries(
		OPEN_LIPSYNC_TO_CUSTOM_VISEME,
	) as Array<[OpenLipSyncVisemeName, string]>) {
		const actual = byLowercase.get(expressionName.toLowerCase());
		if (actual) {
			customVisemes[viseme] = actual;
		}
	}

	const standardMouths: ExpressionBinding<StandardMouthExpression> = {};
	for (const expressionName of STANDARD_MOUTH_EXPRESSIONS) {
		const actual = byLowercase.get(expressionName.toLowerCase());
		if (actual) {
			standardMouths[expressionName] = actual;
		}
	}

	const mouthSourceExpressions = WEBCAM_MOUTH_SOURCE_EXPRESSIONS.flatMap(
		(expressionName) => {
			const actual = byLowercase.get(expressionName.toLowerCase());
			return actual ? [actual] : [];
		},
	);

	return {
		customVisemes,
		standardMouths,
		mouthSourceExpressions,
		customVisemeNames: Object.values(customVisemes),
		standardMouthNames: Object.values(standardMouths),
	};
}

type LipSyncControllerOptions = {
	microphone?: DeviceConfig | null;
	modelPath: string;
	configPath?: string | null;
	smoothing: number;
	routing: VrmExpressionRouting;
};

export class LipSyncController {
	private readonly state: LipSyncState;
	private readonly latestVisemes = zeroVisemeWeights();
	private readonly latestMouth = zeroMouthWeights();
	private readonly routing: VrmExpressionRouting;
	private readonly audioConfig: AudioProcessingConfig;
	private readonly modelIsMultiLabel: boolean;
	private readonly ringBuffer: AudioRingBuffer;
	private readonly melProcessor: MelSpectrogramProcessor;
	private readonly resampler: AudioResampler | null;
	private readonly inputName: string;
	private readonly outputName: string;
	private readonly inferenceBuffer = new Float32Array(
		OPEN_LIPSYNC_VISEMES.length,
	);
	private readonly scratchVisemes = new Float32Array(
		OPEN_LIPSYNC_VISEMES.length,
	);
	private readonly smoothing: number;
	private readonly audioContext: AudioContext;
	private readonly mediaStream: MediaStream;
	private readonly sourceNode: MediaStreamAudioSourceNode;
	private readonly captureNode: AudioWorkletNode;
	private readonly muteNode: GainNode;
	private inferenceChain: Promise<void> = Promise.resolve();

	private constructor(params: {
		audioConfig: AudioProcessingConfig;
		audioContext: AudioContext;
		captureNode: AudioWorkletNode;
		mediaStream: MediaStream;
		melProcessor: MelSpectrogramProcessor;
		modelIsMultiLabel: boolean;
		outputName: string;
		inputName: string;
		resampler: AudioResampler | null;
		ringBuffer: AudioRingBuffer;
		routing: VrmExpressionRouting;
		provider: ProviderName;
		session: ort.InferenceSession;
		smoothing: number;
		sourceNode: MediaStreamAudioSourceNode;
		muteNode: GainNode;
	}) {
		this.audioConfig = params.audioConfig;
		this.audioContext = params.audioContext;
		this.captureNode = params.captureNode;
		this.mediaStream = params.mediaStream;
		this.melProcessor = params.melProcessor;
		this.modelIsMultiLabel = params.modelIsMultiLabel;
		this.outputName = params.outputName;
		this.inputName = params.inputName;
		this.resampler = params.resampler;
		this.ringBuffer = params.ringBuffer;
		this.routing = params.routing;
		this.session = params.session;
		this.smoothing = clamp01(params.smoothing);
		this.sourceNode = params.sourceNode;
		this.muteNode = params.muteNode;
		this.state = {
			enabled: true,
			active: false,
			provider: params.provider,
			customVisemes: params.routing.customVisemeNames,
			standardMouths: params.routing.standardMouthNames,
			visemes: zeroVisemeWeights(),
			mouth: zeroMouthWeights(),
		};
	}

	private readonly session: ort.InferenceSession;

	static async create(
		options: LipSyncControllerOptions,
	): Promise<LipSyncController> {
		const configPath =
			options.configPath ?? resolveSiblingConfigPath(options.modelPath);
		const [modelBuffer, configText, mediaStream] = await Promise.all([
			window.electron.loadBinaryAsset(options.modelPath),
			window.electron.loadTextAsset(configPath),
			openMicrophone(options.microphone),
		]);
		const parsedConfig = JSON.parse(configText) as OpenLipSyncConfig;
		const audioConfig = parseAudioProcessingConfig(parsedConfig);

		const { session, provider } = await createOrtSession(modelBuffer);
		const audioContext = new AudioContext();
		await audioContext.audioWorklet.addModule(
			new URL("./lipsync-worklet.ts", import.meta.url).href,
		);

		const sourceNode = audioContext.createMediaStreamSource(mediaStream);
		const captureNode = new AudioWorkletNode(
			audioContext,
			"otttuber-lipsync-capture",
			{
				numberOfInputs: 1,
				numberOfOutputs: 1,
				outputChannelCount: [1],
			},
		);
		const muteNode = audioContext.createGain();
		muteNode.gain.value = 0;
		sourceNode.connect(captureNode);
		captureNode.connect(muteNode);
		muteNode.connect(audioContext.destination);

		const ringBuffer = new AudioRingBuffer(audioConfig.sampleRate * 3);
		const melProcessor = new MelSpectrogramProcessor(audioConfig);
		const resampler =
			audioContext.sampleRate === audioConfig.sampleRate ?
				null
			:	new AudioResampler(
					audioContext.sampleRate,
					audioConfig.sampleRate,
				);

		const controller = new LipSyncController({
			audioConfig,
			audioContext,
			captureNode,
			mediaStream,
			melProcessor,
			modelIsMultiLabel: parsedConfig.training?.multi_label ?? false,
			outputName: session.outputNames[0] ?? "viseme_logits",
			inputName: session.inputNames[0] ?? "audio_features",
			resampler,
			ringBuffer,
			routing: options.routing,
			provider,
			session,
			smoothing: options.smoothing,
			sourceNode,
			muteNode,
		});

		captureNode.port.onmessage = (event: MessageEvent<Float32Array>) => {
			controller.consumeAudio(event.data);
		};
		await audioContext.resume();
		return controller;
	}

	getState(): LipSyncState {
		return {
			...this.state,
			visemes: { ...this.state.visemes },
			mouth: { ...this.state.mouth },
			customVisemes: [...this.state.customVisemes],
			standardMouths: [...this.state.standardMouths],
		};
	}

	getRouting(): VrmExpressionRouting {
		return this.routing;
	}

	getCustomVisemeWeights(): Partial<Record<OpenLipSyncVisemeName, number>> {
		const weights: Partial<Record<OpenLipSyncVisemeName, number>> = {};
		for (const viseme of OPEN_LIPSYNC_VISEMES) {
			if (this.routing.customVisemes[viseme]) {
				weights[viseme] = this.latestVisemes[viseme];
			}
		}
		return weights;
	}

	getMouthWeights(): WeightMap<StandardMouthExpression> {
		return { ...this.latestMouth };
	}

	private consumeAudio(chunk: Float32Array): void {
		this.state.active = true;
		const samples = this.resampler ? this.resampler.resample(chunk) : chunk;
		if (samples.length === 0) {
			return;
		}
		this.ringBuffer.write(samples);
		for (;;) {
			const features = this.melProcessor.tryProcessNextHop(
				this.ringBuffer,
			);
			if (!features) {
				break;
			}
			this.inferenceChain = this.inferenceChain
				.then(() => this.runInference(features))
				.catch((error) => {
					console.error("Lip-sync inference failed:", error);
				});
		}
	}

	private async runInference(features: Float32Array): Promise<void> {
		const tensor = new ort.Tensor("float32", features, [
			1,
			1,
			features.length,
		]);
		const results = await this.session.run({ [this.inputName]: tensor });
		const outputTensor = results[this.outputName];
		if (!outputTensor) {
			return;
		}

		const logits = outputTensor.data as Float32Array;
		const visemeCount = Math.min(
			logits.length,
			OPEN_LIPSYNC_VISEMES.length,
		);
		if (this.modelIsMultiLabel) {
			for (let i = 0; i < visemeCount; i++) {
				const x = Math.max(-50, Math.min(50, logits[i]));
				this.inferenceBuffer[i] = 1 / (1 + Math.exp(-x));
			}
		} else {
			let maxLogit = -Infinity;
			for (let i = 0; i < visemeCount; i++) {
				maxLogit = Math.max(maxLogit, logits[i]);
			}
			let sum = 0;
			for (let i = 0; i < visemeCount; i++) {
				const value = Math.exp(logits[i] - maxLogit);
				this.inferenceBuffer[i] = value;
				sum += value;
			}
			if (sum > 0) {
				for (let i = 0; i < visemeCount; i++) {
					this.inferenceBuffer[i] /= sum;
				}
			}
		}

		for (let i = visemeCount; i < this.inferenceBuffer.length; i++) {
			this.inferenceBuffer[i] = 0;
		}

		for (let i = 0; i < OPEN_LIPSYNC_VISEMES.length; i++) {
			this.scratchVisemes[i] =
				this.scratchVisemes[i] * this.smoothing +
				this.inferenceBuffer[i] * (1 - this.smoothing);
			this.latestVisemes[OPEN_LIPSYNC_VISEMES[i]] =
				this.scratchVisemes[i];
			this.state.visemes[OPEN_LIPSYNC_VISEMES[i]] =
				this.scratchVisemes[i];
		}

		for (const mouth of STANDARD_MOUTH_EXPRESSIONS) {
			const value = this.latestVisemes[VISEME_TO_STANDARD_MOUTH[mouth]];
			this.latestMouth[mouth] = value;
			this.state.mouth[mouth] = value;
		}
	}

	async dispose(): Promise<void> {
		await this.inferenceChain.catch(() => undefined);
		this.captureNode.port.onmessage = null;
		this.sourceNode.disconnect();
		this.captureNode.disconnect();
		this.muteNode.disconnect();
		for (const track of this.mediaStream.getTracks()) {
			track.stop();
		}
		await this.audioContext.close();
		await this.session.release();
	}
}
