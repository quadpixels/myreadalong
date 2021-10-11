// 2021-10-09
// audio stuff
var audio;
var loudness = 0;
var normalized = [];
var amplitudeSpectrum;
var g_buffer = [];
var g_model;

class LoudnessVis {
  constructor() {
    this.max_value = 16;
    this.x = 500;
    this.y = 575;
    this.w = 100;
    this.h = 13;
  }

  Render(value) {
    while (this.max_value < value) {
      this.max_value += 8;
    }
    const TEXT_SIZE = 16;
    push();
    noStroke();
    fill(32);
    textAlign(LEFT, TOP);
    text("Loudness: " + value.toFixed(1), this.x, this.y + TEXT_SIZE + this.h);
    fill(122);

    rect(this.x, this.y, this.w * value / this.max_value, this.h);
    // xbreaks
    const N = 5;
    textAlign(CENTER, TOP);
    for (let i=0; i<N; i++) {
      const valu = parseInt(this.max_value * i / (N-1));
      const dx = map(i, 0, N-1, this.x, this.x + this.w);
      text(valu+"", dx, this.y+TEXT_SIZE);
      stroke(64);
      line(dx, this.y, dx, this.y + TEXT_SIZE-1);
      noStroke();
    }

    noFill();
    stroke(32);
    rect(this.x, this.y, this.w, this.h);

    pop();
  }
}

class FFTVis {
  constructor() {
    this.nshown = 256;
    this.x = 200;
    this.y = 575;
    this.w = 256;
    this.h = 13;
  }

  myMap(x) {
    return Math.log(x+1) * 14;
  }

  Render(fft) {
    const TEXT_SIZE = 16;
    fill(122);
    for (let i=0; i<this.nshown && i < fft.length; i++) {
      const x0 = map(i, 0, this.nshown-1, this.x, this.x+this.w);
      const x1 = map(i+1,0,this.nshown-1, this.x, this.x+this.w);
      const y0 = constrain(map(this.myMap(fft[i]), 0, 1, this.y+this.h, this.y), this.y, this.y+this.w);
      rect(x0, y0, x1-x0+1, this.h+this.y-y0);
    }

    const nbreaks = 8;
    const fftfreq = audio.meyda._m.sampleRate / 2;
    
    textAlign(CENTER, TOP);
    
    for (let i=0; i<nbreaks; i++) {
      let freq = parseInt(map(i, 0, nbreaks-1, 0, fftfreq));
      if (freq > 1000) {
        freq = parseFloat(freq / 1000).toFixed(1) + "k";
      }
      const dx = map(i, 0, nbreaks-1, this.x, this.x+this.w);
      stroke(122);
      noFill();
      line(dx, this.y, dx, this.y+this.h);

      noStroke();
      fill(122);
      text(freq+"", dx, this.y + this.h + 2);
    }

    const binwidth = parseFloat(fftfreq / fft.length);
    textAlign(LEFT, TOP);
    text(binwidth + " hz * " + fft.length + " bins",
      this.x, this.y + TEXT_SIZE + this.h);

    noFill();
    stroke(32);
    rect(this.x, this.y, this.w, this.h);
  }
}

class AudioStatsViz {
  constructor() {
    this.window_audiosample = new SlidingWindow();
    this.value = 0;
    this.x = 32;
    this.y = 575;
    this.last_ms = 0;
    this.w = 64;
    this.ub = 0; this.lb = 0;
  }

  AddOneEvent() {
    this.window_audiosample.AddOneEvent(millis());
  }

  Render() {
    const TEXT_SIZE = 16
    const ms = millis();
    if (ms > this.last_ms + 1000) {
      this.last_ms = ms;
      this.value = this.window_audiosample.GetCountAfter(ms - 1000);
      this.window_audiosample.RemoveEventsBefore(ms - 1000);
    }
    push();
    noStroke();
    textAlign(LEFT, TOP);
    fill(122);
    text("tfjs " + tf.version.tfjs, this.x, this.y + TEXT_SIZE*3);
    fill(32);
    text(this.value + " audio cb/s", this.x, this.y + TEXT_SIZE*2);
    text(this.lb.toFixed(2) + ".." + this.ub.toFixed(2), this.x, this.y + TEXT_SIZE);

    // draw buffer
    const b = g_buffer.buffer;

    for (let i=0; i<b.length; i++) {
      this.ub = max(this.ub, b[i]);
      this.lb = min(this.lb, b[i]);
    }

    const dy_min = this.y, dy_max = dy_min + 13;
    noFill();
    stroke(122);
    for (let i=1; i<b.length && i < this.w; i++) {
      const idx0 = parseInt(map(i-1, 0, this.w-1, 0, b.length-1));
      const idx1 = parseInt(map(i  , 0, this.w-1, 0, b.length-1));
      const samp0 = b[idx0], samp1 = b[idx1];
      const dy0 = map(samp0, this.lb, this.ub, dy_max, dy_min);
      const dy1 = map(samp1, this.lb, this.ub, dy_max, dy_min);
      const dx0 = i-1+ this.x;
      const dx1 = i  + this.x;
      line(dx0, dy0, dx1, dy1);
    }
    stroke(32);
    rect(this.x, dy_min, this.w, dy_max-dy_min);

    pop();
  }
}

// for manually recording a small segment of sound & testing
// model uses 25ms width and 10ms delta
class RecorderViz {
  constructor() {
    this.Clear();
    this.graph = createGraphics(500, 32);
    this.is_recording = false;
    this.x = 32;
    this.y = 48;
    this.window_delta = 10; // ms
    this.start_rec_millis = 0;
    this.next_expect_millis = 0;
    this.graph.clear();
    this.duration_ms = 0;
    this.px_per_samp = 1;  // 1 sample = 1 px
  }

  Clear() {
    this.buffer = [];
    this.duration_ms = 0;
    if (this.graph != undefined)
      this.graph.clear();
  }

  StartRecording() {
    this.is_recording = true;
    this.start_rec_millis = millis();
    this.next_expect_millis = this.start_rec_millis;
    this.buffer = [];
  }

  myMap(x) {
    return Math.log(x+1) * 14;
  }

  StopRecording() {
    this.is_recording = false;
    
    // Render fft
    const g = this.graph;
    g.clear();
    g.noFill();
    const c0 = color(128, 128, 128);
    const c1 = color(0,   0,   0);
    for (let i=0; i<this.buffer.length && i < g.width; i++) {
      const col = this.buffer[i];
      for (let j=0; j<g.height; j++) {
        const idx = parseInt(j/(g.height-1)*(col.length-1));
        const intensity = constrain(this.myMap(col[idx]), 0, 1);
        g.stroke(lerpColor(c0, c1, intensity));
        g.point(i, g.height - 1 - j);
      }
    }
  }

  AddSpectrumIfRecording(fft, ms) {
    if (!this.is_recording) return;
    while (ms > this.next_expect_millis) {
      this.buffer.push(fft);
      const n = parseInt((ms - this.next_expect_millis) / this.window_delta) + 1;
      this.next_expect_millis += this.window_delta;
      this.duration_ms = ms - this.start_rec_millis;
    }
  }

  Render() {
    push();
    noStroke();
    
    textAlign(LEFT, TOP);

    let dx = 0;
    const txt = "" + this.buffer.length + " | " + 
      (this.duration_ms / 1000).toFixed(1) + "s |";
    dx = textWidth(txt) + 3;

    fill(122);
    text(txt, this.x, this.y);

    if (!this.is_recording) {
      fill(122);
      text("Not recording", this.x + dx, this.y);
    } else {
      fill("#F88");
      text("Recording", this.x + dx, this.y);
    }

    noFill();
    stroke(32);
    const h = this.graph.height;
    let dy = this.y + 15;
    const w = this.buffer.length;
    if (!this.is_recording)
      image(this.graph, this.x, dy);
    noFill();
    stroke(32);
    rect(this.x, dy, w, h);

    pop();
  }
}

class PathfinderViz {
  constructor() {
    this.x = 32;
    this.y = 108;

    this.py2idx = {};
    this.graph = createGraphics(512, 400);
  }

  Render() {
    push();
    noStroke();
    fill(32);
    textAlign(LEFT, TOP);
    text("Pathfinding Viz", this.x, this.y);
    noStroke();
    image(this.graph, this.x, this.y + 15);
    noFill();
    stroke(128);
    rect(this.x, this.y + 15, this.graph.width, this.graph.height);
    pop();
  }

  MapColor(val) {
    if (val == 0) { val = -99; }
    else { val = Math.log(val); }
    val = constrain(val, -20, 0);
    val = map(val, -20, 0, 0, 1);
    
    const c0 = color(128, 128, 128);
    const c1 = color(0,   0,   0);

    return lerpColor(c0, c1, val);
  }

  RenderPredictionOutput(o) {
    // O is a tensor
    const len = o.shape[1];
    const vocab_size = o.shape[2];
    const g = this.graph;
    g.clear();
    g.noFill();
    for (let i=0; i<len && i < g.width; i++) {
      const line = o.slice([0,i,0], [1,1,vocab_size]).dataSync();
      for (let j=0; j<g.height; j++) {
        let lb = parseInt(map(j, 0, g.height-1, 0, vocab_size-1));
        let ub = parseInt(map(j+1,0,g.height-1, 0, vocab_size-1));
        if (ub < lb) ub = lb;

        let s = 0;
        for (let k=lb; k<=ub; k++) {
          s += line[k];
        }

        g.stroke(this.MapColor(s));
        g.point(i, j);
      }
    }
  }
}

var g_loudness_vis, g_fft_vis;
var g_recording = false;
//var g_rec_mfcc = [];
//var graph_rec_mfcc;
var graph_mfcc0, graph_diff;

var soundReady = false;
var classifying = false;

var normalized = [];
var currentPrediction = "";
var rotation = 0.0;

let g_frame_count = 0;
let g_audiostats_viz = new AudioStatsViz();

let temp0;

function setup() {
  createCanvas(640, 640);
  frameRate(60);
  //graph_rec_mfcc = createGraphics(512, 16);
  //graph_mfcc0 = createGraphics(512, 16);
  graph_diff = createGraphics(512, 512);
  audio = new MicrophoneInput(512);
  g_loudness_vis = new LoudnessVis();
  g_fft_vis = new FFTVis();

  const b = createButton("Load model");
  b.position(16, 16);
  b.mousePressed(async () => {
    g_model = await LoadModel();
  })

  const b1 = createButton("Predict");
  b1.position(106, 16);
  b1.mousePressed(async() => {
    console.log(g_recorderviz.buffer)
    temp0 = await DoPrediction(g_recorderviz.buffer);
    g_pathfinder_viz.RenderPredictionOutput(temp0);
  })
}

function DrawMFCC(mfcc, g) {
  g.clear();
  g.push();
  g.strokeWeight(1);
  const W = mfcc.length;
  const H = mfcc[0].length;
  
  const from = color(218, 165, 32);
  const to = color(72, 61, 139);
  
  for (let x=0; x<W; x++) {
    for (let y=0; y<H; y++) {
      const val = mfcc[x][y];
      //const norm_val = map(val, -10, 30, 0, 1);
      g.stroke(lerpColor(from, to, val));
      //g.stroke(color(norm_val*255, norm_val*255, 255));
      g.point(x, y);
    }
  }
  g.pop();
}

function DrawDiffMatrix(diffs, g) {
  g.clear();
  g.push();
  const H = diffs.length;
  const W = diffs[0].length;
  const from = color(218, 165, 32);
  const to = color(72, 61, 139);
  for (let y=0; y<H; y++) {
    for (let x=0; x<W; x++) {
      const val = diffs[y][x];
      g.stroke(lerpColor(from, to, val));
      g.point(x, y);
    }
  }
  g.pop();
}

function draw() {
  const ms = millis();
  if (g_frame_count == 0) {
  //  DrawMFCC(MFCC0, graph_mfcc0);
    g_recorderviz = new RecorderViz();
    g_pathfinder_viz = new PathfinderViz();
  }
  
  background(255);
  textSize(12);
  
  // cost matrix
  if(0) {
    push();
    translate(16, 32);
    rotate(PI/2);
    image(graph_rec_mfcc, 0, 0);
    pop();
    push();
    translate(32, 16);
    image(graph_mfcc0, 0, 0);
    pop();
    
    image(graph_diff, 32, 32);
  }
  
  push();
  
  if (soundReady) {
    fill(0);
    noStroke();

    g_loudness_vis.Render(loudness.total);
    g_fft_vis.Render(amplitudeSpectrum);
    textAlign(LEFT, TOP);
    
    if (g_recording) {
      fill(0, 0, 255);
      noStroke();
      text("REC " + g_rec_mfcc.length, 16, 16);
    }
    
    g_audiostats_viz.Render();
    g_recorderviz.Render();
    g_pathfinder_viz.Render();
  }
  
  pop();
  
  
  g_frame_count ++;
}


function soundDataCallback(soundData) {
  const ms = millis();
  g_audiostats_viz.AddOneEvent();
  soundReady = true;
  //mfcc = soundData.mfcc;
  amplitudeSpectrum = soundData.amplitudeSpectrum;
  loudness = soundData.loudness;
  g_buffer = soundData;
  g_recorderviz.AddSpectrumIfRecording(amplitudeSpectrum, ms);

  //for (var i = 0; i < 13; i++) {
  //  normalized[i] = map(mfcc[i], -10, 30, 0, 1);
  //}
  
  //g_rec_mfcc.push(mfcc.slice());
}

function keyPressed() {
  if (key == 'r') {
    g_recorderviz.StartRecording();
  } else if (key == 'p') { // Print recorded
    console.log(JSON.stringify(g_rec_mfcc));
  }
}

function keyReleased() {
  if (key == 'r') {
    g_recorderviz.StopRecording();
  }
}