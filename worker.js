// worker.js (corregido: usa ImageData -> cv.matFromImageData en lugar de cv.imread con HTML elements)
self.importScripts('https://docs.opencv.org/4.x/opencv.js');

let cvReady = false;
let orb = null, bf = null;
let templMat = null, templGray = null, templKeypoints = null, templDescriptors = null;
let prevGray = null, prevPts = null, templPts = null;
let procW = 160, procH = 120;
let MODE = 'detection'; // 'detection' | 'tracking'
let lastTrackTime = 0;

let MATCH_RATIO = 0.9;
let MAX_GOOD_MATCHES = 150;
let minMatchCount = 6;

self.Module = self.Module || {};
self.Module.onRuntimeInitialized = () => {
  cvReady = true;
  postMessage({type:'ready'});
};

// safe delete helper
function safeDelete(m){
  try{ if (m && typeof m.delete === 'function') m.delete(); }catch(e){}
}

// --- NEW: convertir ImageBitmap -> cv.Mat (grayscale) sin usar cv.imread(HTMLImageElement)
// dibuja ImageBitmap en OffscreenCanvas, obtiene ImageData y usa cv.matFromImageData
function bitmapToGrayMat(bitmap, w = procW, h = procH){
  // create offscreen canvas sized to procW x procH
  const oc = new OffscreenCanvas(w, h);
  const ctx = oc.getContext('2d');
  ctx.drawImage(bitmap, 0, 0, w, h);
  // obtener ImageData
  const id = ctx.getImageData(0, 0, w, h);
  // convertir a Mat RGBA
  const mat = cv.matFromImageData(id); // mat tipo CV_8UC4
  const gray = new cv.Mat();
  cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
  mat.delete();
  return gray; // caller must delete
}

// --- NEW: carga template desde URL -> cv.Mat usando fetch + createImageBitmap + matFromImageData
async function loadImageToMat(url, maxSize=800){
  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const blob = await resp.blob();
    const imgBitmap = await createImageBitmap(blob);
    // calcular tamaño target (mantener aspect)
    let tw = imgBitmap.width, th = imgBitmap.height;
    if (Math.max(tw, th) > maxSize){
      const s = maxSize / Math.max(tw, th);
      tw = Math.round(tw * s);
      th = Math.round(th * s);
    }
    // dibujar en OffscreenCanvas a tamaño tw x th
    const oc = new OffscreenCanvas(tw, th);
    const ctx = oc.getContext('2d');
    ctx.drawImage(imgBitmap, 0, 0, tw, th);
    const id = ctx.getImageData(0,0,tw,th);
    const matRGBA = cv.matFromImageData(id); // CV_8UC4
    const mat = new cv.Mat();
    cv.cvtColor(matRGBA, mat, cv.COLOR_RGBA2GRAY); // keep grayscale template
    matRGBA.delete();
    try{ imgBitmap.close(); }catch(e){}
    return mat; // caller deletes
  } catch(err){
    postMessage({type:'error', msg:'loadImageToMat error: '+err});
    throw err;
  }
}

// ORB detect+compute helper
function detectAndCompute(grayMat){
  const kps = new cv.KeyPointVector();
  const desc = new cv.Mat();
  try {
    orb.detect(grayMat, kps);
    orb.compute(grayMat, kps, desc);
  } catch(e){
    try { orb.detectAndCompute(grayMat, new cv.Mat(), kps, desc); } catch(err){ postMessage({type:'log', msg:'detectAndCompute failed: '+err}); }
  }
  return {kps, desc};
}

// knn + ratio test
function knnGoodMatches(des1, des2, ratio=0.9, maxMatches=150){
  const good = [];
  let matches = new cv.DMatchVectorVector();
  try { bf.knnMatch(des1, des2, matches, 2); }
  catch(e){ postMessage({type:'log', msg:'knnMatch error: '+e}); try{ matches.delete(); }catch(_){ } return good; }
  for (let i=0;i<matches.size();i++){
    const mv = matches.get(i);
    if (mv.size() >= 2){
      const m = mv.get(0), n = mv.get(1);
      if (typeof m.distance !== 'undefined' && typeof n.distance !== 'undefined'){
        if (m.distance <= ratio * n.distance){
          good.push({queryIdx: m.queryIdx, trainIdx: m.trainIdx, distance: m.distance});
        }
      }
    } else if (mv.size() === 1){
      const m = mv.get(0);
      if (typeof m.distance !== 'undefined' && m.distance < 40) good.push({queryIdx: m.queryIdx, trainIdx: m.trainIdx, distance: m.distance});
    }
    try{ mv.delete(); }catch(e){}
    if (good.length >= maxMatches) break;
  }
  try{ matches.delete(); }catch(e){}
  return good;
}

function computeHomographyFromMatches(goodMatches, frameKps, templKps){
  const src = []; const dst = [];
  for (let i=0;i<goodMatches.length;i++){
    const gm = goodMatches[i];
    const q = gm.queryIdx, t = gm.trainIdx;
    const kpf = frameKps.get(q);
    const kpt = templKps.get(t);
    src.push(kpf.pt.x, kpf.pt.y);
    dst.push(kpt.pt.x, kpt.pt.y);
  }
  const srcMat = cv.matFromArray(goodMatches.length, 1, cv.CV_32FC2, src);
  const dstMat = cv.matFromArray(goodMatches.length, 1, cv.CV_32FC2, dst);
  const mask = new cv.Mat();
  let H = null;
  try {
    H = cv.findHomography(srcMat, dstMat, cv.RANSAC, 5.0, mask);
  } catch(e){
    postMessage({type:'log', msg:'findHomography failed: '+e});
  }
  let inliers = 0;
  if (mask && !mask.isDeleted()){
    for (let i=0;i<mask.rows;i++) if (mask.ucharPtr(i,0)[0]) inliers++;
  }
  try{ srcMat.delete(); dstMat.delete(); }catch(e){}
  return {H, mask, inliers};
}

function homographyToCorners(H){
  if (!H) return null;
  try {
    const H_inv = new cv.Mat();
    cv.invert(H, H_inv, cv.DECOMP_LU);
    const tw = templGray.cols, th = templGray.rows;
    const tplCorners = cv.matFromArray(4,1,cv.CV_32FC2, [0,0, tw,0, tw,th, 0,th]);
    const dstCorners = new cv.Mat();
    cv.perspectiveTransform(tplCorners, dstCorners, H_inv);
    const out = [];
    for (let i=0;i<4;i++){
      out.push(dstCorners.floatAt(i,0), dstCorners.floatAt(i,1));
    }
    tplCorners.delete(); dstCorners.delete(); H_inv.delete();
    return out;
  } catch(e){
    postMessage({type:'log', msg:'homographyToCorners err: '+e});
    return null;
  }
}

// init template (uses new loadImageToMat)
async function initTemplate(url){
  safeDelete(templMat); safeDelete(templGray); safeDelete(templKeypoints); safeDelete(templDescriptors);
  templGray = await loadImageToMat(url, 800);
  if (!templGray) throw new Error('templGray null');
  // apply CLAHE if available (try/catch)
  try {
    const clahe = new cv.CLAHE(2.0, new cv.Size(8,8));
    const tmp = new cv.Mat();
    clahe.apply(templGray, tmp);
    templGray.delete();
    templGray = tmp;
    clahe.delete();
    postMessage({type:'log', msg:'CLAHE applied to template'});
  } catch(e){ /* ignore */ }

  // create templMat just for dimensions if needed (we can derive cols/rows from templGray)
  templMat = new cv.Mat();
  cv.cvtColor(templGray, templMat, cv.COLOR_GRAY2RGBA); // a CV_8UC4 mat if ever needed
  // init ORB and BF
  try { orb = new cv.ORB(600, 1.2, 8, 31, 0, 2, cv.ORB_HARRIS_SCORE, 31, 20); } catch(e){ orb = new cv.ORB(); }
  bf = new cv.BFMatcher(cv.NORM_HAMMING, false);

  templKeypoints = new cv.KeyPointVector();
  templDescriptors = new cv.Mat();
  try { orb.detect(templGray, templKeypoints); orb.compute(templGray, templKeypoints, templDescriptors); }
  catch(e){
    try { orb.detectAndCompute(templGray, new cv.Mat(), templKeypoints, templDescriptors); } catch(err){ postMessage({type:'error', msg:'ORB detect failed: '+err}); }
  }

  postMessage({type:'log', msg:`Template loaded KP:${templKeypoints.size()} desc:${templDescriptors.rows}`});
}

// main processing of ImageBitmap (now using bitmapToGrayMat)
async function processFrameBitmap(bitmap){
  if (!cvReady || !templGray || !orb){
    try{ bitmap.close(); }catch(e){}
    postMessage({type:'result', matches:0, inliers:0, corners:null});
    return;
  }

  const grayMat = bitmapToGrayMat(bitmap, procW, procH);
  try{ bitmap.close(); }catch(e){}

  // tracking path
  if (MODE === 'tracking' && prevGray && prevPts && templPts){
    try {
      const nextPts = new cv.Mat();
      const status = new cv.Mat();
      const err = new cv.Mat();
      cv.calcOpticalFlowPyrLK(prevGray, grayMat, prevPts, nextPts, status, err, new cv.Size(21,21), 3);

      const goodNext = []; const goodTempl = [];
      for (let i=0;i<status.rows;i++){
        if (status.ucharPtr(i,0)[0] === 1){
          const nx = nextPts.floatAt(i,0); const ny = nextPts.floatAt(i,1);
          const tx = templPts.floatAt(i,0); const ty = templPts.floatAt(i,1);
          if (!isFinite(nx) || !isFinite(ny)) continue;
          goodNext.push(nx, ny); goodTempl.push(tx, ty);
        }
      }
      nextPts.delete(); status.delete(); err.delete();

      if (goodNext.length/2 < 6){
        MODE = 'detection';
        safeDelete(prevPts); safeDelete(prevGray); safeDelete(templPts);
        prevPts = null; prevGray = null; templPts = null;
        postMessage({type:'log', msg:'tracking lost -> detection'});
      } else {
        const n = goodNext.length/2;
        const nextMat = cv.matFromArray(n, 1, cv.CV_32FC2, goodNext);
        const templSub = cv.matFromArray(n, 1, cv.CV_32FC2, goodTempl);
        const mask = new cv.Mat();
        let H = null;
        try { H = cv.findHomography(nextMat, templSub, cv.RANSAC, 5.0, mask); } catch(e){ postMessage({type:'log', msg:'findHomography(flow) err: '+e}); }
        let inliers = 0;
        for (let i=0;i<mask.rows;i++) if (mask.ucharPtr(i,0)[0]) inliers++;
        if (H && !H.empty() && inliers >= Math.max(4, Math.floor(n * 0.25))){
          const corners = homographyToCorners(H);
          // build filtered prevPts & templPts from mask
          const goodNextFiltered = []; const goodTemplFiltered = [];
          for (let i=0;i<mask.rows;i++){
            if (mask.ucharPtr(i,0)[0]){
              goodNextFiltered.push(goodNext[2*i], goodNext[2*i+1]);
              goodTemplFiltered.push(goodTempl[2*i], goodTempl[2*i+1]);
            }
          }
          safeDelete(prevPts); safeDelete(prevGray);
          prevPts = cv.matFromArray(goodNextFiltered.length/2,1,cv.CV_32FC2, goodNextFiltered);
          templPts = cv.matFromArray(goodTemplFiltered.length/2,1,cv.CV_32FC2, goodTemplFiltered);
          prevGray = grayMat.clone();
          lastTrackTime = performance.now();
          postMessage({type:'result', matches:n, inliers, corners});
          // cleanup
          try{ nextMat.delete(); templSub.delete(); mask.delete(); if (H && !H.isDeleted) H.delete(); }catch(e){}
          return;
        } else {
          // fallback to detection; cleanup temporals
          try{ nextMat.delete(); templSub.delete(); mask.delete(); if (H && !H.isDeleted) H.delete(); }catch(e){}
          // continue to detection
        }
      }
    } catch(err){
      postMessage({type:'log', msg:'optical flow exception: '+err});
      MODE = 'detection';
      safeDelete(prevPts); safeDelete(prevGray); safeDelete(templPts);
      prevPts = null; prevGray = null; templPts = null;
    }
  }

  // DETECTION: ORB + knn + homography
  try {
    const det = detectAndCompute(grayMat);
    const frameKps = det.kps; const frameDesc = det.desc;
    const tplK = templKeypoints ? templKeypoints.size() : 0;
    const frmK = frameKps ? frameKps.size() : 0;

    if (!tplK || !frmK){
      safeDelete(frameKps); safeDelete(frameDesc); safeDelete(grayMat);
      postMessage({type:'result', matches:0, inliers:0, corners:null});
      return;
    }

    const goodMatches = knnGoodMatches(frameDesc, templDescriptors, MATCH_RATIO, MAX_GOOD_MATCHES);
    if (goodMatches.length < minMatchCount){
      safeDelete(frameKps); safeDelete(frameDesc); safeDelete(grayMat);
      postMessage({type:'result', matches:goodMatches.length, inliers:0, corners:null});
      return;
    }

    const {H, mask, inliers} = computeHomographyFromMatches(goodMatches, frameKps, templKeypoints);
    let corners = null;
    if (H && !H.empty() && inliers >= Math.max(4, Math.floor(goodMatches.length * 0.2))){
      corners = homographyToCorners(H);

      // build arrays for prevPts & templPts using goodMatches & mask
      const framePtsArr = [];
      const templPtsArr = [];
      for (let idx=0; idx<goodMatches.length; idx++){
        if (mask.ucharPtr(idx,0)[0]){
          const gm = goodMatches[idx];
          const q = gm.queryIdx, t = gm.trainIdx;
          const kpf = frameKps.get(q);
          const kpt = templKeypoints.get(t);
          framePtsArr.push(kpf.pt.x, kpf.pt.y);
          templPtsArr.push(kpt.pt.x, kpt.pt.y);
        }
      }

      safeDelete(prevPts); safeDelete(prevGray); safeDelete(templPts);
      if (framePtsArr.length/2 >= 6){
        prevPts = cv.matFromArray(framePtsArr.length/2, 1, cv.CV_32FC2, framePtsArr);
        templPts = cv.matFromArray(templPtsArr.length/2, 1, cv.CV_32FC2, templPtsArr);
        prevGray = grayMat.clone();
        MODE = 'tracking';
        lastTrackTime = performance.now();
      } else {
        MODE = 'detection';
      }

      postMessage({type:'result', matches:goodMatches.length, inliers, corners});
    } else {
      postMessage({type:'result', matches:goodMatches.length, inliers:inliers||0, corners:null});
    }

    safeDelete(frameKps); safeDelete(frameDesc);
    // if we didn't keep grayMat for tracking, delete it
    if (!(MODE === 'tracking' && prevGray)) safeDelete(grayMat);
    return;
  } catch(err){
    postMessage({type:'log', msg:'detection error: '+err});
    safeDelete(grayMat);
    postMessage({type:'result', matches:0, inliers:0, corners:null});
    return;
  }
}

// message handling
self.onmessage = async (ev) => {
  const d = ev.data;
  if (d.type === 'init'){
    // wait cv runtime
    if (!cvReady){
      let waited = 0;
      while(!cvReady && waited < 8000){ await new Promise(r=>setTimeout(r,100)); waited+=100; }
      if (!cvReady){ postMessage({type:'error', msg:'OpenCV runtime not initialized'}); return; }
    }
    procW = d.procW || procW; procH = d.procH || procH;
    try {
      await initTemplate(d.targetUrl);
      postMessage({type:'log', msg:'template initialized in worker'});
    } catch(e){
      postMessage({type:'error', msg:'initTemplate failed: '+e});
    }
    return;
  } else if (d.type === 'resize'){
    procW = d.procW; procH = d.procH;
    postMessage({type:'log', msg:`worker resized to ${procW}x${procH}`});
    return;
  } else if (d.type === 'frame'){
    // frame is transferred as bitmap (transfer ownership)
    const bitmap = d.bitmap || ev.data.bitmap;
    if (!bitmap){
      postMessage({type:'log', msg:'no bitmap in message'}); return;
    }
    await processFrameBitmap(bitmap);
  }
};
