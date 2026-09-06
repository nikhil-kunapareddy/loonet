import { BookOpen, ChevronRight, History, LoaderCircle, Menu, ShieldCheck, Sparkles, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { DetectionResult } from './components/DetectionResult'
import { ImageUploader } from './components/ImageUploader'
import { analyzeImage } from './services/detectionService'
import { clearHistory, loadHistory, saveHistory } from './services/historyService'
import type { DetectionResult as Result } from './types/detection'

 type Page = 'check' | 'results' | 'about'

function formatDate(timestamp: string) {
  return new Date(timestamp).toLocaleDateString(undefined, { month: 'long', day: 'numeric', year: 'numeric' })
}

function App() {
  const [page, setPage] = useState<Page>('check')
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<Result | null>(null)
  const [history, setHistory] = useState<Result[]>(loadHistory)
  const [processing, setProcessing] = useState(false)
  const [saved, setSaved] = useState(false)
  const [mobileMenu, setMobileMenu] = useState(false)

  useEffect(() => () => { if (previewUrl?.startsWith('blob:')) URL.revokeObjectURL(previewUrl) }, [previewUrl])

  function selectFile(nextFile: File) {
    if (previewUrl?.startsWith('blob:')) URL.revokeObjectURL(previewUrl)
    setFile(nextFile)
    setPreviewUrl(URL.createObjectURL(nextFile))
    setResult(null)
    setSaved(false)
  }

  function removeFile() {
    if (previewUrl?.startsWith('blob:')) URL.revokeObjectURL(previewUrl)
    setFile(null)
    setPreviewUrl(null)
  }

  async function imageToDataUrl(nextFile: File) {
    return new Promise<string>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(String(reader.result))
      reader.onerror = reject
      reader.readAsDataURL(nextFile)
    })
  }

  async function checkImage() {
    if (!file || !previewUrl) return
    setProcessing(true)
    const dataUrl = await imageToDataUrl(file)
    const nextResult = await analyzeImage(file, dataUrl)
    setResult(nextResult)
    setProcessing(false)
    setPage('check')
  }

  function resetCheck() {
    removeFile()
    setResult(null)
    setSaved(false)
    setPage('check')
  }

  function saveCurrentResult() {
    if (!result) return
    setHistory(saveHistory(result))
    setSaved(true)
  }

  function openResult(nextResult: Result) {
    setResult(nextResult)
    setFile(null)
    setPreviewUrl(null)
    setSaved(true)
    setPage('check')
  }

  function navigate(nextPage: Page) {
    setPage(nextPage)
    setMobileMenu(false)
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="header-inner">
          <button className="brand" onClick={() => navigate('check')} aria-label="Go to check an image">
            <span className="brand-mark"><span /></span>
            <span><strong>National Loon Center</strong><small>Conservation field tools</small></span>
          </button>
          <nav className={`main-nav ${mobileMenu ? 'nav-open' : ''}`} aria-label="Main navigation">
            <button className={page === 'check' ? 'nav-link active' : 'nav-link'} onClick={() => navigate('check')}>Check an image</button>
            <button className={page === 'results' ? 'nav-link active' : 'nav-link'} onClick={() => navigate('results')}><History size={16} /> Previous checks</button>
            <button className={page === 'about' ? 'nav-link active' : 'nav-link'} onClick={() => navigate('about')}><BookOpen size={16} /> About</button>
          </nav>
          <button className="menu-button" onClick={() => setMobileMenu(!mobileMenu)} aria-label="Toggle navigation">{mobileMenu ? <X size={21} /> : <Menu size={21} />}</button>
        </div>
      </header>

      <main className="main-content">
        {page === 'check' && !result && <>
          <div className="page-intro">
            <div className="intro-copy"><p className="eyebrow"><span className="eyebrow-line" /> Field image check</p><h1>Is this a <em>loon?</em></h1><p className="intro-lede">Upload a photo or take a picture to check whether a loon is present.</p></div>
            <div className="intro-stamp"><ShieldCheck size={17} /><span>Built for careful<br />conservation work</span></div>
          </div>
          <div className="check-panel">
            <ImageUploader file={file} previewUrl={previewUrl} onFile={selectFile} onRemove={removeFile} />
            {file && <button className="button button-accent check-button" onClick={checkImage} disabled={processing}>{processing ? <><LoaderCircle className="spin" size={18} /> Checking image...</> : <>Check for loons <ChevronRight size={18} /></>}</button>}
            {processing && <div className="processing-message" role="status"><Sparkles size={17} /><span><strong>We're looking closely.</strong><small>Looking for loon-like features in this photo.</small></span></div>}
          </div>
          <div className="small-note"><span className="note-rule" /> AI-generated results can contain errors. Review results when accuracy is important.</div>
        </>}

        {page === 'check' && result && <DetectionResult result={result} onCheckAnother={resetCheck} onSave={saveCurrentResult} saved={saved} />}

        {page === 'results' && <section className="history-page"><div className="section-heading"><div><p className="eyebrow">Your field notes</p><h1>Previous checks</h1><p className="intro-lede">Review images you've checked with the loon detector.</p></div>{history.length > 0 && <button className="text-button danger-text" onClick={() => { if (window.confirm('Clear saved checks? Demo samples will stay available.')) { clearHistory(); setHistory(loadHistory()) } }}>Clear history</button>}</div>{history.length === 0 ? <div className="empty-history"><History size={29} /><h2>No saved checks yet</h2><p>Results you save will appear here for easy review.</p><button className="button button-dark" onClick={() => navigate('check')}>Check an image</button></div> : <div className="history-grid">{history.map((item) => <button className="history-card" key={item.id} onClick={() => openResult(item)}><img src={item.imageUrl} alt="Saved loon check" /><span className="history-card-content">{item.isSample && <span className="sample-label">Demo sample</span>}<span className={item.detections.length ? 'history-status found' : 'history-status'}>{item.detections.length ? 'Loon detected' : 'No loon detected'}</span><strong>{item.detections.length ? `${item.detections.length} ${item.detections.length === 1 ? 'loon' : 'loons'} · ${Math.round(Math.max(...item.detections.map((d) => d.confidence)) * 100)}% highest confidence` : 'Review this result'}</strong><small>{formatDate(item.timestamp)}</small></span><ChevronRight size={18} /></button>)}</div>}</section>}

        {page === 'about' && <section className="about-page"><div className="about-hero"><p className="eyebrow">A tool for the field</p><h1>Built to support<br /><em>loon research.</em></h1><p className="intro-lede">This tool uses computer vision to help identify loons in photographs. It is designed to support researchers and conservation teams working with loon populations and habitat.</p></div><div className="about-content"><div><p className="eyebrow">The simple version</p><h2>How it works</h2></div><ol className="steps"><li><b>01</b><span>Upload or take a photo.</span></li><li><b>02</b><span>The system analyzes the image.</span></li><li><b>03</b><span>If a loon is detected, we highlight it.</span></li><li><b>04</b><span>Review the result with your own expertise.</span></li></ol></div><div className="important-note"><ShieldCheck size={22} /><div><strong>Important note</strong><p>This tool is designed to assist researchers, not replace expert judgment. AI-generated results can contain errors and should be reviewed when accuracy is important.</p></div></div></section>}
      </main>
      <footer className="site-footer"><span>National Loon Center</span><span>Prototype for conservation research</span></footer>
    </div>
  )
}

export default App
