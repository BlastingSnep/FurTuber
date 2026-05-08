/// <reference types="vite/client" />

interface AppConfig {
	camera: {
		position: [number, number, number];
		lookAt: [number, number, number];
		fov: number;
	};
	webcam?: {
		deviceLabel?: string | null;
		deviceId?: string | null;
	};
	microphone?: {
		deviceLabel?: string | null;
		deviceId?: string | null;
	};
	model: {
		path: string;
		scale: number;
		rotation: [number, number, number];
		mirror: boolean;
	};
	tracking: {
		blendshapeAmplify: Record<string, number>;
		blendshapeFilter: { minCutoff: number; beta: number };
		blendshapeFilterOverrides: Record<
			string,
			{ minCutoff: number; beta: number }
		>;
		headFilter: { minCutoff: number; beta: number };
		armCalibration?: {
			poseScale?: { x: number; y: number; z: number };
			minCutoff?: number;
			beta?: number;
		};
		handFilter?: { minCutoff?: number; beta?: number };
		lipsync?: {
			enabled?: boolean;
			modelPath?: string;
			configPath?: string | null;
			smoothing?: number;
			audioBlendWeight?: number;
			webcamBlendWeight?: number;
		};
	};
}

interface DebugData {
	detected: boolean;
	blendshapes: Array<{ name: string; value: number }>;
	head: { pitch: number; yaw: number; roll: number };
	arms: Array<{ name: string; value: number }>;
	visemes: Array<{ name: string; value: number }>;
	mouth: Array<{ name: string; value: number }>;
	lipsync: {
		enabled: boolean;
		active: boolean;
		provider: string | null;
		customVisemes: string[];
		standardMouths: string[];
	};
}

interface Window {
	electron: {
		loadVrm(filename: string): Promise<ArrayBuffer>;
		loadConfig(): Promise<AppConfig | null>;
		loadBinaryAsset(relativePath: string): Promise<ArrayBuffer>;
		loadTextAsset(relativePath: string): Promise<string>;
		sendDebugData(data: DebugData): void;
		onDebugData(callback: (data: DebugData) => void): void;
	};
}
