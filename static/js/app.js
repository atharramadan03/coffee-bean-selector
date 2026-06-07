// Coffee Bean Selector AI — app.js

let uploadedFile=null, cameraBlob=null, cameraStream=null, currentTab='upload';
const HIST_KEY = 'cbai_v3';

const EMOJI = { Dark:'☕', Green:'🌱', Light:'🫖', Medium:'☕' };
const BADGE = { Dark:'cls-dark', Green:'cls-green', Light:'cls-light', Medium:'cls-medium' };
const INFO  = {
  Dark:   { desc:'Biji kopi sangrai gelap dengan rasa bold, pahit kuat, dan aroma smoky. Proses sangrai panjang menghasilkan kadar kafein lebih rendah.', chars:['Warna coklat sangat gelap hingga hitam','Permukaan berminyak','Rasa pahit dominan, low acidity','Kadar kafein lebih rendah'], brew:'Espresso · Americano · Cold Brew' },
  Green:  { desc:'Biji kopi mentah yang belum melalui proses sangrai. Kaya antioksidan dan asam klorogenat alami yang bermanfaat bagi kesehatan.', chars:['Warna hijau keabuan','Tekstur keras dan padat','Aroma segar seperti rumput','Kandungan antioksidan tinggi'], brew:'Green Coffee Extract · Suplemen' },
  Light:  { desc:'Biji kopi sangrai ringan dengan rasa fruity, keasaman cerah, dan aroma floral yang kuat. Kadar kafein lebih tinggi karena proses sangrai singkat.', chars:['Warna coklat terang','Permukaan kering (tidak berminyak)','Rasa asam & fruity','Aroma floral yang kuat'], brew:'Pour Over · V60 · Chemex · Aeropress' },
  Medium: { desc:'Biji kopi sangrai sedang dengan keseimbangan sempurna antara rasa, keasaman, dan aroma. Pilihan paling populer dan universal.', chars:['Warna coklat sedang','Permukaan sedikit berminyak','Rasa seimbang dan smooth','Kafein moderat'], brew:'Drip Coffee · French Press · Cappuccino' }
};

function goTo(id) { document.getElementById(id)?.scrollIntoView({behavior:'smooth',block:'start'}); }

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab-btn').forEach((el,i) => el.classList.toggle('active',(i===0&&tab==='upload')||(i===1&&tab==='camera')));
  document.getElementById('uploadPanel').classList.toggle('active', tab==='upload');
  document.getElementById('cameraPanel').classList.toggle('active', tab==='camera');
  updateBtn();
}

function handleFileSelect(e) { const f=e.target.files[0]; if(f) loadFile(f); }

function loadFile(file) {
  if (!file.type.startsWith('image/')) return toast('File harus berupa gambar!','error');
  if (file.size > 10*1024*1024) return toast('Ukuran file maks 10MB!','error');
  uploadedFile = file;
  const r = new FileReader();
  r.onload = e => {
    document.getElementById('uploadPreview').src = e.target.result;
    document.getElementById('uploadPreviewWrap').classList.add('show');
    document.getElementById('dropZone').style.display = 'none';
  };
  r.readAsDataURL(file);
  updateBtn();
  toast('Gambar siap dianalisis!','success');
}

function clearUpload() {
  uploadedFile = null;
  document.getElementById('fileInput').value = '';
  document.getElementById('uploadPreview').src = '';
  document.getElementById('uploadPreviewWrap').classList.remove('show');
  document.getElementById('dropZone').style.display = '';
  updateBtn();
  resetResult();
}

function handleDragOver(e) { e.preventDefault(); document.getElementById('dropZone').classList.add('dragover'); }
function handleDragLeave() { document.getElementById('dropZone').classList.remove('dragover'); }
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('dropZone').classList.remove('dragover');
  const f = e.dataTransfer.files[0]; if(f) loadFile(f);
}

async function startCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment',width:{ideal:640},height:{ideal:480}}});
    const v = document.getElementById('cameraVideo');
    v.srcObject = cameraStream; v.style.display = 'block';
    document.getElementById('camEmpty').style.display = 'none';
    document.getElementById('camReticle').style.display = 'block';
    document.getElementById('btnOpenCam').disabled = true;
    document.getElementById('btnCapture').disabled = false;
    document.getElementById('btnStopCam').disabled = false;
    toast('Kamera aktif!','success');
  } catch(e) { toast('Tidak dapat mengakses kamera.','error'); }
}

function capturePhoto() {
  const v=document.getElementById('cameraVideo'), c=document.getElementById('cameraCanvas');
  c.width=v.videoWidth; c.height=v.videoHeight;
  c.getContext('2d').drawImage(v,0,0);
  c.toBlob(blob => {
    cameraBlob=blob;
    document.getElementById('camPreview').src=c.toDataURL('image/jpeg',0.93);
    document.getElementById('camPreviewWrap').classList.add('show');
    updateBtn(); toast('Foto diambil!','success');
  },'image/jpeg',0.93);
}

function stopCamera() {
  if(cameraStream){cameraStream.getTracks().forEach(t=>t.stop());cameraStream=null;}
  document.getElementById('cameraVideo').style.display='none';
  document.getElementById('camEmpty').style.display='';
  document.getElementById('camReticle').style.display='none';
  document.getElementById('btnOpenCam').disabled=false;
  document.getElementById('btnCapture').disabled=true;
  document.getElementById('btnStopCam').disabled=true;
}

function clearCamera() {
  cameraBlob=null;
  document.getElementById('camPreview').src='';
  document.getElementById('camPreviewWrap').classList.remove('show');
  updateBtn();
  resetResult();
}

function resetResult() {
  const resultSec = document.getElementById('resultSection');
  const placeholder = document.getElementById('resultPlaceholder');
  resultSec.classList.remove('show');
  if (placeholder) placeholder.style.display = '';
}

function updateBtn() {
  const has=(currentTab==='upload'&&uploadedFile)||(currentTab==='camera'&&cameraBlob);
  document.getElementById('btnAnalyze').disabled=!has;
}

async function runPrediction() {
  const loading=document.getElementById('loadingWrap');
  const resultSec=document.getElementById('resultSection');
  const placeholder=document.getElementById('resultPlaceholder');
  document.getElementById('btnAnalyze').disabled=true;
  loading.classList.add('show');
  resultSec.classList.remove('show');
  if(placeholder) placeholder.style.display='none';
  try {
    let response;
    if(currentTab==='upload'&&uploadedFile){
      const fd=new FormData(); fd.append('image',uploadedFile);
      response=await fetch('/api/predict',{method:'POST',body:fd});
    } else if(currentTab==='camera'&&cameraBlob){
      const b64=await new Promise(res=>{const r=new FileReader();r.onload=e=>res(e.target.result);r.readAsDataURL(cameraBlob);});
      response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_base64:b64})});
    } else { toast('Pilih gambar terlebih dahulu!','error'); return; }
    const result=await response.json();
    if(!response.ok||!result.success) throw new Error(result.error||'Prediksi gagal');
    showResult(result.data);
    saveHistory(result.data);
  } catch(err) {
    toast('Prediksi gagal: '+err.message,'error');
    if(placeholder) placeholder.style.display='';
  } finally { loading.classList.remove('show'); updateBtn(); }
}

function showResult(data) {
  const klass=data.predicted_class, info=INFO[klass]||{};
  document.getElementById('resultIcon').textContent=EMOJI[klass]||'☕';
  document.getElementById('resultKlass').textContent=klass+' Roast';
  animateCount(document.getElementById('confNum'),0,data.confidence,1200);
  setTimeout(()=>{document.getElementById('confFill').style.width=data.confidence+'%';},80);
  document.getElementById('metaTime').textContent=data.inference_time_ms+' ms';
  document.getElementById('metaDate').textContent=data.timestamp;
  const fn=data.filename||'—';
  document.getElementById('metaFile').textContent=fn.length>20?fn.slice(0,18)+'…':fn;

  const container=document.getElementById('probBars');
  container.innerHTML='';
  data.all_classes.forEach(cls=>{
    const isTop=cls.class===klass;
    const row=document.createElement('div');
    row.className='prob-row'+(isTop?' is-top':'');
    row.innerHTML=`<div class="prob-cls">${EMOJI[cls.class]||''} ${cls.class}</div>
      <div class="prob-bar-track"><div class="prob-bar-fill" id="pb_${cls.class}"></div></div>
      <div class="prob-pct">${cls.probability}%</div>`;
    container.appendChild(row);
  });
  setTimeout(()=>{ data.all_classes.forEach(cls=>{ const el=document.getElementById('pb_'+cls.class); if(el) el.style.width=cls.probability+'%'; }); },150);

  document.getElementById('infoTitle').textContent=(EMOJI[klass]||'')+' '+klass+' Roast';
  document.getElementById('infoDesc').textContent=info.desc||'';
  const ul=document.getElementById('infoChars'); ul.innerHTML='';
  (info.chars||[]).forEach(c=>{ const li=document.createElement('li'); li.textContent=c; ul.appendChild(li); });
  document.getElementById('infoBrew').textContent='Cocok untuk: '+(info.brew||'');
  document.getElementById('resultSection').classList.add('show');
}

function animateCount(el,from,to,dur) {
  const s=performance.now();
  (function step(now){ const t=Math.min((now-s)/dur,1),e=1-Math.pow(1-t,3);
    el.innerHTML=(from+(to-from)*e).toFixed(1)+'<sub>%</sub>';
    if(t<1) requestAnimationFrame(step);
  })(performance.now());
}

function getHist() { try{return JSON.parse(localStorage.getItem(HIST_KEY)||'[]');}catch{return[];} }

function saveHistory(data) {
  const h=getHist();
  h.unshift({filename:data.filename||'—',predicted_class:data.predicted_class,confidence:data.confidence,timestamp:data.timestamp});
  if(h.length>50) h.splice(50);
  localStorage.setItem(HIST_KEY,JSON.stringify(h));
  renderHistory();
}

function renderHistory() {
  const h=getHist();
  const empty=document.getElementById('historyEmpty');
  const tbl=document.getElementById('historyTableWrap');
  const tbody=document.getElementById('historyTbody');
  if(!empty||!tbl||!tbody) return;
  if(h.length===0){empty.style.display='';tbl.style.display='none';return;}
  empty.style.display='none'; tbl.style.display='block';
  tbody.innerHTML='';
  h.forEach((item,i)=>{
    const tr=document.createElement('tr');
    tr.style.animationDelay=(i*25)+'ms';
    tr.innerHTML=`<td style="color:var(--ink-faint);font-weight:600">${i+1}</td>
      <td style="max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${item.filename}">${item.filename}</td>
      <td><span class="cls-badge ${BADGE[item.predicted_class]||''}">${EMOJI[item.predicted_class]||''} ${item.predicted_class}</span></td>
      <td><span class="conf-val">${item.confidence}%</span></td>
      <td style="font-size:0.74rem;color:var(--ink-faint)">${item.timestamp}</td>`;
    tbody.appendChild(tr);
  });
}

function clearHistory() {
  showConfirmDialog(
    'Hapus Riwayat?',
    'Semua riwayat prediksi akan dihapus permanen. Tindakan ini tidak dapat dibatalkan.',
    () => {
      localStorage.removeItem(HIST_KEY);
      renderHistory();
      toast('Riwayat dihapus.','success');
    }
  );
}

function showConfirmDialog(title, message, onConfirm) {
  // Remove existing dialog if any
  const existing = document.getElementById('customDialog');
  if (existing) existing.remove();

  const overlay = document.createElement('div');
  overlay.id = 'customDialog';
  overlay.className = 'dialog-overlay';
  overlay.innerHTML = `
    <div class="dialog-box animate-in">
      <div class="dialog-icon"><i class="bi bi-trash3-fill"></i></div>
      <div class="dialog-title">${title}</div>
      <div class="dialog-msg">${message}</div>
      <div class="dialog-actions">
        <button class="dialog-btn-cancel" id="dialogCancel">Batal</button>
        <button class="dialog-btn-confirm" id="dialogConfirm">
          <i class="bi bi-trash3"></i> Hapus Semua
        </button>
      </div>
    </div>`;

  document.body.appendChild(overlay);

  // Trigger animation
  requestAnimationFrame(() => overlay.classList.add('show'));

  const close = () => {
    overlay.classList.remove('show');
    setTimeout(() => overlay.remove(), 250);
  };

  document.getElementById('dialogCancel').onclick = close;
  document.getElementById('dialogConfirm').onclick = () => { close(); onConfirm(); };
  overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
  document.addEventListener('keydown', function esc(e) {
    if (e.key === 'Escape') { close(); document.removeEventListener('keydown', esc); }
  });
}

function toast(msg,type='info') {
  const stack=document.getElementById('toastStack');
  const icons={success:'bi-check-circle-fill',error:'bi-exclamation-circle-fill',info:'bi-info-circle-fill'};
  const el=document.createElement('div');
  el.className=`toast ${type}`;
  el.innerHTML=`<i class="bi ${icons[type]||icons.info}"></i>${msg}`;
  stack.appendChild(el);
  setTimeout(()=>el.remove(),3500);
}

function initCanvas() {
  const cvs=document.getElementById('heroCanvas');
  if(!cvs) return;
  const ctx=cvs.getContext('2d');
  const resize=()=>{ cvs.width=cvs.offsetWidth; cvs.height=cvs.offsetHeight; };
  resize(); window.addEventListener('resize',resize);
  const pts=Array.from({length:50},()=>({
    x:Math.random()*cvs.width, y:Math.random()*cvs.height,
    r:Math.random()*1.8+0.3, vx:(Math.random()-.5)*.25,
    vy:-(Math.random()*.4+.1), a:Math.random()*.35+.05
  }));
  (function draw() {
    ctx.clearRect(0,0,cvs.width,cvs.height);
    pts.forEach(p=>{
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle=`rgba(212,168,83,${p.a})`; ctx.fill();
      p.x+=p.vx; p.y+=p.vy; p.a-=.0007;
      if(p.y<-8||p.a<=0){p.x=Math.random()*cvs.width;p.y=cvs.height+5;p.a=Math.random()*.35+.05;p.r=Math.random()*1.8+.3;}
    });
    requestAnimationFrame(draw);
  })();
}

document.addEventListener('DOMContentLoaded',()=>{
  renderHistory(); initCanvas();
  fetch('/api/health').then(r=>r.json()).then(d=>{
    if(d.model_loaded) toast('Model AI berhasil dimuat ✓','success');
    else toast('Model AI belum dimuat!','error');
  }).catch(()=>{});
});