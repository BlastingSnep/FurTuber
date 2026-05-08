import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("electron", {
	loadVrm: (filename: string): Promise<ArrayBuffer> =>
		ipcRenderer.invoke("load-vrm", filename),
	loadConfig: (): Promise<unknown> => ipcRenderer.invoke("load-config"),
	loadBinaryAsset: (relativePath: string): Promise<ArrayBuffer> =>
		ipcRenderer.invoke("load-binary-asset", relativePath),
	loadTextAsset: (relativePath: string): Promise<string> =>
		ipcRenderer.invoke("load-text-asset", relativePath),
	sendDebugData: (data: unknown): void =>
		ipcRenderer.send("debug-data", data),
	onDebugData: (callback: (data: unknown) => void): void => {
		ipcRenderer.on("debug-data", (_event, data) => callback(data));
	},
});
