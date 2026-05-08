import { app, BrowserWindow, ipcMain, globalShortcut } from "electron";
import { dirname, join } from "path";
import { readFileSync } from "fs";

let debugWin: BrowserWindow | null = null;

function readRelativeBinaryAsset(relativePath: string): ArrayBuffer {
	const resolvedPath = join(app.getAppPath(), relativePath);
	const buffer = readFileSync(resolvedPath);
	return buffer.buffer.slice(
		buffer.byteOffset,
		buffer.byteOffset + buffer.byteLength,
	);
}

function readRelativeTextAsset(relativePath: string): string {
	return readFileSync(join(app.getAppPath(), relativePath), "utf-8");
}

function createDebugWindow(): void {
	debugWin = new BrowserWindow({
		width: 480,
		height: 900,
		title: "OttTuber Debug",
		backgroundColor: "#0d1117",
		webPreferences: {
			preload: join(__dirname, "../preload/index.js"),
			contextIsolation: true,
			nodeIntegration: false,
		},
	});

	if (process.env["ELECTRON_RENDERER_URL"]) {
		debugWin.loadURL(`${process.env["ELECTRON_RENDERER_URL"]}/debug.html`);
	} else {
		debugWin.loadFile(join(__dirname, "../renderer/debug.html"));
	}

	debugWin.on("closed", () => {
		debugWin = null;
	});
}

function createWindow(): void {
	const win = new BrowserWindow({
		width: 800,
		height: 900,
		transparent: true,
		frame: false,
		backgroundColor: "#00000000",
		webPreferences: {
			preload: join(__dirname, "../preload/index.js"),
			// Needed so the renderer can fetch MediaPipe WASM from CDN and open file:// VRM paths
			webSecurity: false,
			contextIsolation: true,
			nodeIntegration: false,
		},
	});

	// Grant webcam access without prompting
	win.webContents.session.setPermissionRequestHandler(
		(_webContents, permission, callback) => {
			callback(permission === "media");
		},
	);

	if (process.env["ELECTRON_RENDERER_URL"]) {
		win.loadURL(process.env["ELECTRON_RENDERER_URL"]);
	} else {
		win.loadFile(join(__dirname, "../renderer/index.html"));
	}
}

app.whenReady().then(() => {
	ipcMain.handle("load-vrm", (_event, relativePath: string) => {
		return readRelativeBinaryAsset(relativePath);
	});

	ipcMain.handle("load-config", () => {
		try {
			return JSON.parse(readRelativeTextAsset("config.json"));
		} catch {
			return null;
		}
	});

	ipcMain.handle("load-binary-asset", (_event, relativePath: string) => {
		try {
			return readRelativeBinaryAsset(relativePath);
		} catch (error) {
			const message =
				error instanceof Error ? error.message : String(error);
			throw new Error(
				`Failed to load binary asset "${relativePath}" from "${dirname(
					join(app.getAppPath(), relativePath),
				)}": ${message}`,
			);
		}
	});

	ipcMain.handle("load-text-asset", (_event, relativePath: string) => {
		try {
			return readRelativeTextAsset(relativePath);
		} catch (error) {
			const message =
				error instanceof Error ? error.message : String(error);
			throw new Error(
				`Failed to load text asset "${relativePath}": ${message}`,
			);
		}
	});

	// Forward debug data from the renderer to the debug window
	ipcMain.on("debug-data", (_event, data) => {
		debugWin?.webContents.send("debug-data", data);
	});

	createWindow();
	createDebugWindow();

	// Ctrl+Shift+D toggles the debug window
	globalShortcut.register("CommandOrControl+Shift+D", () => {
		if (debugWin) {
			debugWin.close();
		} else {
			createDebugWindow();
		}
	});

	app.on("activate", () => {
		if (BrowserWindow.getAllWindows().length === 0) createWindow();
	});
});

app.on("window-all-closed", () => {
	if (process.platform !== "darwin") app.quit();
});

app.on("will-quit", () => {
	globalShortcut.unregisterAll();
});
