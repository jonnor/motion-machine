import { html, render, useState, useEffect, useCallback } from 'https://esm.sh/htm/preact/standalone'

// ── API Abstraction ────────────────────────────────────────────────────────────

class FileSystemAPI {
    async list()                        { throw new Error('Not implemented') }
    async read(name)                    { throw new Error('Not implemented') }
    async write(name, file, onProgress) { throw new Error('Not implemented') }
    async remove(name)                  { throw new Error('Not implemented') }
}

// ── LocalStorage implementation (demo) ────────────────────────────────────────

const LS_KEY = 'filebrowser_demo'

class LocalStorageAPI extends FileSystemAPI {
    _load() {
        try { return JSON.parse(localStorage.getItem(LS_KEY) || '{}') }
        catch { return {} }
    }
    _save(store) {
        localStorage.setItem(LS_KEY, JSON.stringify(store))
    }

    async list() {
        const store = this._load()
        return Object.entries(store).map(([name, { size, modified }]) => ({
            name, type: 'file', size, modified
        }))
    }

    async read(name) {
        const store = this._load()
        if (!store[name]) throw new Error(`Not found: ${name}`)
        return new Blob([store[name].content], { type: 'application/octet-stream' })
    }

    async write(name, file, onProgress) {
        const text = await file.text()
        // Simulate chunked progress
        for (let p = 0; p <= 100; p += 20) {
            onProgress(p)
            await new Promise(r => setTimeout(r, 40))
        }
        const store = this._load()
        store[name] = { content: text, size: file.size, modified: Date.now() }
        this._save(store)
    }

    async remove(name) {
        const store = this._load()
        if (!store[name]) throw new Error(`Not found: ${name}`)
        delete store[name]
        this._save(store)
    }
}

// ── Remote (MicroDot) implementation ──────────────────────────────────────────

class RemoteAPI extends FileSystemAPI {
    constructor(base = '/files') {
        super()
        this.base = base.replace(/\/$/, '')
    }

    async list() {
        const res = await fetch(this.base)
        if (!res.ok) throw new Error(`List failed: ${res.status}`)
        const data = await res.json()
        return data.files || []
    }

    async read(name) {
        const res = await fetch(`${this.base}/${name}`)
        if (!res.ok) throw new Error(`Download failed: ${res.status}`)
        return res.blob()
    }

    async write(name, file, onProgress) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest()
            xhr.open('PUT', `${this.base}/${name}`)
            xhr.upload.onprogress = e => {
                if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100))
            }
            xhr.onload = () => xhr.status < 300 ? resolve() : reject(new Error(`Upload failed: ${xhr.status}`))
            xhr.onerror = () => reject(new Error('Network error'))
            xhr.send(file)
        })
    }

    async remove(name) {
        const res = await fetch(`${this.base}/${name}`, { method: 'DELETE' })
        if (!res.ok) throw new Error(`Delete failed: ${res.status}`)
    }
}

// ── Switch: change this line to swap implementations ──────────────────────────

const USE_DEMO = false   // false → RemoteAPI talking to your MicroDot device

const api = USE_DEMO
    ? new LocalStorageAPI()
    : new RemoteAPI('/files')

// ── Download helper ────────────────────────────────────────────────────────────

async function downloadBlob(blob, name) {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = name; a.click()
    URL.revokeObjectURL(url)
}

// ── Icons ──────────────────────────────────────────────────────────────────────

const Icon = ({ d, size = 16 }) => html`
    <svg width=${size} height=${size} viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d=${d}/>
    </svg>`

const Icons = {
    file:     'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z M14 2v6h6',
    folder:   'M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z',
    download: 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M7 10l5 5 5-5 M12 15V3',
    trash:    'M3 6h18 M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6 M10 11v6 M14 11v6 M9 6V4h6v2',
    upload:   'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M17 8l-5-5-5 5 M12 3v12',
    refresh:  'M23 4v6h-6 M1 20v-6h6 M3.51 9a9 9 0 0 1 14.85-3.36L23 10 M1 14l4.64 4.36A9 9 0 0 0 20.49 15',
    alert:    'M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z M12 9v4 M12 17h.01',
    chip:     'M9 3H5a2 2 0 0 0-2 2v4m6-6h6m-6 0v18m6-18h4a2 2 0 0 1 2 2v4m0 0h-18m18 0v6m0 0h-18m18 0v4a2 2 0 0 1-2 2h-4m0 0H9m6 0v-18',
}

// ── Components ─────────────────────────────────────────────────────────────────

const StatusBar = ({ message, type }) => !message ? null : html`
    <div class="status-bar ${type}">
        ${type === 'error' && html`<${Icon} d=${Icons.alert} size=${14}/>`}
        <span>${message}</span>
    </div>`

const ProgressBar = ({ value }) => value === null ? null : html`
    <div class="progress-track">
        <div class="progress-fill" style=${{ width: `${value}%` }}></div>
    </div>`

function formatSize(bytes) {
    if (bytes == null) return ''
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / 1048576).toFixed(1)} MB`
}

function FileRow({ file, onDownload, onDelete }) {
    const [confirming, setConfirming] = useState(false)

    const handleDelete = () => {
        if (confirming) { onDelete(file.name); setConfirming(false) }
        else { setConfirming(true); setTimeout(() => setConfirming(false), 3000) }
    }

    return html`
        <div class="file-row">
            <span class="file-icon">
                <${Icon} d=${file.type === 'dir' ? Icons.folder : Icons.file} size=${14}/>
            </span>
            <span class="file-name">${file.name}</span>
            <span class="file-meta">${formatSize(file.size)}</span>
            <div class="file-actions">
                ${file.type !== 'dir' && html`
                    <button class="btn-icon" title="Download" onClick=${() => onDownload(file.name)}>
                        <${Icon} d=${Icons.download} size=${14}/>
                    </button>`}
                <button class="btn-icon ${confirming ? 'danger' : ''}"
                        title=${confirming ? 'Confirm?' : 'Delete'}
                        onClick=${handleDelete}>
                    <${Icon} d=${Icons.trash} size=${14}/>
                    ${confirming && html`<span class="confirm-label">confirm?</span>`}
                </button>
            </div>
        </div>`
}

function UploadZone({ onUpload }) {
    const [dragging, setDragging] = useState(false)
    const handleDrop = e => {
        e.preventDefault(); setDragging(false)
        const file = e.dataTransfer.files[0]
        if (file) onUpload(file)
    }
    const handleChange = e => {
        if (e.target.files[0]) onUpload(e.target.files[0])
        e.target.value = ''
    }
    return html`
        <div class="upload-zone ${dragging ? 'dragging' : ''}"
             onDragOver=${e => { e.preventDefault(); setDragging(true) }}
             onDragLeave=${() => setDragging(false)}
             onDrop=${handleDrop}>
            <${Icon} d=${Icons.upload} size=${18}/>
            <span>Drop file or <label class="upload-link">
                browse<input type="file" hidden onChange=${handleChange}/>
            </label></span>
        </div>`
}

function FileBrowser() {
    const [files, setFiles]       = useState([])
    const [loading, setLoading]   = useState(false)
    const [progress, setProgress] = useState(null)
    const [status, setStatus]     = useState({ message: '', type: '' })

    const notify      = (message, type = 'info') => setStatus({ message, type })
    const clearStatus = () => setStatus({ message: '', type: '' })

    const refresh = useCallback(async () => {
        setLoading(true); clearStatus()
        try { setFiles(await api.list()) }
        catch (e) { notify(e.message, 'error') }
        finally { setLoading(false) }
    }, [])

    useEffect(() => { refresh() }, [])

    const handleDownload = async name => {
        try { await downloadBlob(await api.read(name), name); notify(`Downloaded ${name}`) }
        catch (e) { notify(e.message, 'error') }
    }

    const handleUpload = async file => {
        clearStatus(); setProgress(0)
        try { await api.write(file.name, file, setProgress); notify(`Uploaded ${file.name}`); await refresh() }
        catch (e) { notify(e.message, 'error') }
        finally { setProgress(null) }
    }

    const handleDelete = async name => {
        try { await api.remove(name); notify(`Deleted ${name}`); await refresh() }
        catch (e) { notify(e.message, 'error') }
    }

    return html`
        <div class="fb-root">
            <header class="fb-header">
                <div class="fb-title">
                    <${Icon} d=${Icons.chip} size=${13}/>
                    <span>FILE SYSTEM</span>
                </div>
                <div class="fb-header-right">
                    ${USE_DEMO && html`<div class="demo-badge">DEMO · localStorage</div>`}
                    <button class="btn-icon ${loading ? 'spinning' : ''}" onClick=${refresh} title="Refresh">
                        <${Icon} d=${Icons.refresh} size=${15}/>
                    </button>
                </div>
            </header>

            <${StatusBar} message=${status.message} type=${status.type}/>
            <${ProgressBar} value=${progress}/>

            <div class="file-list">
                ${files.length === 0 && !loading && html`
                    <div class="empty">No files · upload one below</div>`}
                ${files.map(f => html`
                    <${FileRow} key=${f.name} file=${f}
                        onDownload=${handleDownload}
                        onDelete=${handleDelete}/>`)}
            </div>

            <${UploadZone} onUpload=${handleUpload}/>
        </div>`
}

// ── Styles ─────────────────────────────────────────────────────────────────────

const CSS = `
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg:      #0f0f0f;
        --surface: #161616;
        --border:  #272727;
        --text:    #d4d0c8;
        --muted:   #555;
        --accent:  #e8a020;
        --danger:  #c0392b;
        --success: #2ecc71;
        --font:    'IBM Plex Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        background: var(--bg); color: var(--text);
        font-family: var(--font); font-size: 13px;
        min-height: 100vh;
        display: flex; align-items: flex-start; justify-content: center;
        padding: 48px 16px;
    }
    .fb-root { width: 100%; max-width: 660px; border: 1px solid var(--border); background: var(--surface); }
    .fb-header {
        display: flex; align-items: center; justify-content: space-between;
        padding: 10px 14px; border-bottom: 1px solid var(--border); background: var(--bg);
    }
    .fb-title { display: flex; align-items: center; gap: 8px; color: var(--accent); font-weight: 500; letter-spacing: 0.12em; font-size: 11px; }
    .fb-header-right { display: flex; align-items: center; gap: 8px; }
    .demo-badge { font-size: 9px; letter-spacing: 0.1em; color: var(--muted); border: 1px solid var(--border); padding: 2px 6px; }
    .status-bar { display: flex; align-items: center; gap: 6px; padding: 6px 14px; font-size: 11px; border-bottom: 1px solid var(--border); }
    .status-bar.error { color: var(--danger); background: #1c0a0a; }
    .status-bar.info  { color: var(--success); }
    .progress-track { height: 2px; background: var(--border); }
    .progress-fill { height: 100%; background: var(--accent); transition: width 0.1s linear; }
    .file-list { min-height: 60px; }
    .file-row {
        display: grid; grid-template-columns: 20px 1fr 64px auto;
        align-items: center; gap: 8px; padding: 8px 14px;
        border-bottom: 1px solid var(--border); transition: background 0.1s;
    }
    .file-row:last-child { border-bottom: none; }
    .file-row:hover { background: rgba(255,255,255,0.02); }
    .file-icon { color: var(--muted); display: flex; }
    .file-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .file-meta { color: var(--muted); font-size: 11px; text-align: right; }
    .file-actions { display: flex; gap: 4px; justify-content: flex-end; }
    .confirm-label { font-size: 10px; margin-left: 2px; }
    .empty { padding: 28px; text-align: center; color: var(--muted); font-size: 11px; letter-spacing: 0.08em; }
    .btn-icon {
        display: inline-flex; align-items: center; gap: 3px;
        background: none; border: 1px solid transparent; color: var(--muted);
        cursor: pointer; padding: 4px 6px; font-family: var(--font); font-size: 11px;
        transition: color 0.12s, border-color 0.12s;
    }
    .btn-icon:hover  { color: var(--text); border-color: var(--border); }
    .btn-icon.danger { color: var(--danger); border-color: var(--danger); }
    .btn-icon.spinning svg { animation: spin 1s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .upload-zone {
        display: flex; align-items: center; justify-content: center; gap: 10px;
        padding: 18px; border-top: 1px solid var(--border);
        color: var(--muted); font-size: 12px; transition: background 0.15s, color 0.15s;
    }
    .upload-zone.dragging { background: rgba(232,160,32,0.05); color: var(--accent); }
    .upload-link { color: var(--accent); cursor: pointer; text-decoration: underline; text-underline-offset: 2px; }
`

// ── Mount ──────────────────────────────────────────────────────────────────────

function mount(selector = '#app') {
    const style = document.createElement('style')
    style.textContent = CSS
    document.head.appendChild(style)
    const root = document.querySelector(selector)
    if (!root) throw new Error(`No element found: ${selector}`)
    render(html`<${FileBrowser}/>`, root)
}

export { mount, FileBrowser, LocalStorageAPI, RemoteAPI }
