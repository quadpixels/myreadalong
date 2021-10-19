// 2021-10-09
// audio stuff
var normalized = [];
var amplitudeSpectrum;
var g_buffer = [];
var g_model;

const STATUS_Y = 185;
// Audio processor
var g_my_processor;

class LoudnessVis {
  constructor() {
    this.max_value = 16;
    this.x = 500;
    this.y = STATUS_Y;
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
    this.nshown = 200;
    this.x = 240;
    this.y = STATUS_Y;
    this.w = 256;
    this.h = 13;

    this.fft = [];
    this.sliding_window = new SlidingWindow();
    this.last_win_ms = 0;
    this.fft_per_sec = 0;
  }

  myMap(x) {
    let ret = map(log(x), 1, 12, 0, 1);
    ret = constrain(ret, 0, 1);
    return ret;
  }

  AddOneEntry(fft) {
    this.fft = fft;
    this.sliding_window.AddOneEvent(millis());
  }

  Render() {
    const ms = millis();
    if (ms >= this.last_win_ms + 1000) {
      this.fft_per_sec = this.sliding_window.GetCountAndTotalWeightAfter(ms - 1000);
      this.sliding_window.RemoveEventsBefore(ms - 1000);
      this.last_win_ms += parseInt((ms - this.last_win_ms) / 1000) * 1000;
    }

    const fft = this.fft;
    const TEXT_SIZE = 16;
    fill(122);
    for (let i=0; i<this.nshown && i < fft.length; i++) {
      const x0 = map(i, 0, this.nshown-1, this.x, this.x+this.w);
      const x1 = map(i+1,0,this.nshown-1, this.x, this.x+this.w);
      const y0 = constrain(map(this.myMap(fft[i]), 0, 1, this.y+this.h, this.y), this.y, this.y+this.w);
      rect(x0, y0, x1-x0+1, this.h+this.y-y0);
    }

    const nbreaks = 9;
    
    // Resampled to 16KHz, so max nyquist frequency is 8Khz
    const fftfreq = 8000;
    
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
      text(freq+"", dx, this.y + TEXT_SIZE);
    }

    const binwidth = parseFloat(fftfreq / (fft.length / 2));
    textAlign(LEFT, TOP);
    text(binwidth + " hz * " + fft.length + " bins, showing " + this.nshown + " bins",
      this.x, this.y + TEXT_SIZE*2);
    fill(32);
    text(this.fft_per_sec[0] + " ffts/s ", this.x, this.y + TEXT_SIZE * 3);

    noFill();
    stroke(32);
    rect(this.x, this.y, this.w, this.h);
  }
}

class AudioStatsViz {
  constructor() {
    this.window_audiosample = new SlidingWindow();
    this.samp_per_sec = 0;
    this.cb_per_sec   = 0;
    this.x = 32;
    this.y = STATUS_Y;
    this.last_ms = 0;
    this.w = 64;
    this.ub = 0; this.lb = 0;
  }

  AddOneEvent(buffer) {
    const ms = millis();
    this.window_audiosample.AddOneEvent(millis(), buffer.length);
  }

  Render() {
    const TEXT_SIZE = 16
    const ms = millis();
    if (ms > this.last_ms + 1000) {
      const x = this.window_audiosample.GetCountAndTotalWeightAfter(ms - 1000);
      this.samp_per_sec = x[1];
      this.cb_per_sec   = x[0];
      this.window_audiosample.RemoveEventsBefore(ms - 1000);

      this.last_ms += parseInt((ms - this.last_ms) / 1000) * 1000;
      //this.last_ms += 1000;
    }
    push();
    noStroke();
    textAlign(LEFT, TOP);
    
    //fill(122);
    //text("tfjs " + tf.version.tfjs, this.x, this.y + TEXT_SIZE*4);

    fill(122);
    text(this.lb.toFixed(2) + ".." + this.ub.toFixed(2), this.x, this.y + TEXT_SIZE);
    fill(32);
    text(this.samp_per_sec + " sp/s", this.x, this.y + TEXT_SIZE*2);
    text(this.cb_per_sec + " cb/s", this.x, this.y + TEXT_SIZE*3);
    // draw buffer
    const b = g_buffer;

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
    this.graph.clear();
    this.start_record_ms = 0;
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
    this.buffer = [];
    this.start_record_ms = millis();
  }

  myMap(x) {
    let ret = map(Math.log(x+1), 0, 20, 0, 1);
    ret = constrain(ret, 0, 1);
    return ret;
  }

  StopRecording() {
    this.is_recording = false;
    this.duration_ms = millis() - this.start_record_ms;
    
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

  AddSpectrumIfRecording(fft) {
    if (!this.is_recording) return;
    this.buffer.push(fft);
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
      this.duration_ms = millis() - this.start_record_ms;
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

    this.result = "x";
    this.predict_time = 0;
    this.decode_time = 0;
  }

  SetResult(res, pred_time, dec_time) {
    this.result = res;
    this.predict_time = pred_time;
    this.decode_time = dec_time;
  }

  Render() {
    const TEXT_SIZE = 15;
    push();
    noStroke();
    fill(32);
    textAlign(LEFT, TOP);

    //text("Result panel", this.x, this.y);
    fill(128);
    text("tfjs " + tf.version.tfjs, this.x, this.y);
    //image(this.graph, this.x, this.y + TEXT_SIZE);
    
    fill(32);
    if (this.result != "") {
      text("Result: " + this.result, this.x, this.y + TEXT_SIZE);
      text("Predict time: " + this.predict_time + " ms", this.x, this.y + TEXT_SIZE*2);
      text("Decode time: " + this.decode_time + " ms", this.x, this.y + TEXT_SIZE*3);
    }

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

    if (true) {
      let pylist = [];
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

        // Get argmax
        let maxprob = -1e20, maxidx = -999;
        for (let j=0; j<line.length; j++) {
          if (line[j] > maxprob) {
            maxprob = line[j];
            maxidx = j;
          }
        }
        if (maxidx != -999) {
          pylist.push(PINYIN_LIST[maxidx]);
        } else {
          pylist.push(" ");
        }
        console.log(pylist)
      }
    }
  }
}

var g_loudness_vis, g_fft_vis;
var g_recording = false;
//var g_rec_mfcc = [];
//var graph_rec_mfcc;
var graph_mfcc0, graph_diff;

var soundReady = true;

var normalized = [];
var currentPrediction = "";
var rotation = 0.0;

let g_frame_count = 0;
let g_input_audio_stats_viz = new AudioStatsViz();
let g_downsp_audio_stats_viz = new AudioStatsViz();

let temp0, temp1;
let temp0array;
let g_textarea;
let g_audio;

let g_audio_file_input;
let g_audio_elt;

async function setup() {
  g_audio_file_input = document.getElementById("audio_input");
  g_audio_file_input.addEventListener("input", async (x) => {
    console.log("addEventListener");
    console.log(x.target.files);
    let the_file;
    x.target.files.forEach((f) => {
      the_file = f;
    })
    if (the_file) {
      g_audio_elt = createElement("audio");  // p5.js-wrapped object
      g_audio_elt.attribute("controls", "")
      g_audio_elt.elt.src = URL.createObjectURL(the_file);
      g_audio_elt.position(280, 16);

      console.log(the_file);
      
      // create some audio context
      let audio_context = new AudioContext();
      let source = audio_context.createMediaElementSource(g_audio_elt.elt);
      let m = await audio_context.audioWorklet.addModule('my-processor.js');
      const myProcessor = await CreateMyProcessor(audio_context);
      source.connect(myProcessor);
    }
  });

  createCanvas(640, 640);
  frameRate(60);

  graph_diff = createGraphics(512, 512);
  g_loudness_vis = new LoudnessVis();
  g_fft_vis = new FFTVis();

  let b0, b2;

  b0 = createButton("Mic");
  b0.position(16, 16);
  b0.mousePressed(() => {
    g_audio = new MicrophoneInput(512);
    b2.elt.disabled = true;
    b0.elt.disabled = true;
  });

  b2 = createButton("File");
  b2.position(56, 16);
  b2.mousePressed(() => {
    g_audio_file_input.click();
    b2.elt.disabled = true;
    b0.elt.disabled = true;
  });

  let b, b1;
  b = createButton("Load model");
  b.position(112, 16);
  b.mousePressed(async () => {
    g_model = await LoadModel();
    b1.elt.disabled = false;
  })

  b1 = createButton("Predict");
  b1.position(206, 16);
  b1.elt.disabled = true;
  b1.mousePressed(async() => {
    console.log(g_recorderviz.buffer)
    const ms0 = millis();
    temp0 = await DoPrediction(g_recorderviz.buffer);
    const ms1 = millis();
    //g_pathfinder_viz.RenderPredictionOutput(temp0);
    temp0array = []; // for ctc
    const T = temp0.shape[1];
    const S = temp0.shape[2];
    for (let t=0; t<T; t++) {
      let line = [];
      let src = temp0.slice([0, t, 0], [1, 1, S]).dataSync();
      for (let s=0; s<S; s++) {
        line.push(src[s]);
      }
      temp0array.push(line);
    }
    let blah = Decode(temp0array, 10, S-1);
    let out = ""
    blah[0].forEach((x) => {
      out = out + PINYIN_LIST[x] + " "
    });
    const ms2 = millis();
    g_pathfinder_viz.SetResult(out, ms1-ms0, ms2-ms1);
    console.log(out);
  });

  g_textarea = createElement("textarea", "");
  g_textarea.size(320, 50);
  g_textarea.position(32, STATUS_Y + 100)
  g_textarea.hide();

  g_downsp_audio_stats_viz.x = 130;
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

    //g_loudness_vis.Render(loudness.total);
    g_fft_vis.Render();
    textAlign(LEFT, TOP);
    
    if (g_recording) {
      fill(0, 0, 255);
      noStroke();
      text("REC " + g_rec_mfcc.length, 16, 16);
    }
    
    g_input_audio_stats_viz.Render();
    g_downsp_audio_stats_viz.Render();
    g_recorderviz.Render();
    g_pathfinder_viz.Render();
  }

  pop();

  g_frame_count ++;
}

function SoundDataCallbackMyAnalyzer(buffer, downsampled, fft_frames) {
  soundReady = true;
  g_input_audio_stats_viz.AddOneEvent(buffer);
  g_buffer = buffer;
  g_downsp_audio_stats_viz.AddOneEvent(downsampled);
}

function soundDataCallback(soundData) {
  const ms = millis();
  if (g_input_audio_stats_viz == undefined) return;
  if (g_recorderviz == undefined) return;
  soundReady = true;
  //mfcc = soundData.mfcc;
  
  g_buffer = soundData;
  g_recorderviz.AddSpectrumIfRecording(amplitudeSpectrum, ms);
}

function keyPressed() {
  if (key == 'r') {
    g_recorderviz.StartRecording();
  } else if (key == 'p') { // Print recorded
    g_textarea.show();
    const x = g_recorderviz.buffer;
    let txt = "";
    x.forEach((line) => {
      for (let i=0; i<line.length; i++) {
        if (i>0) {
          txt += ","
        }
        txt += ScaleFFTDataPoint(line[i])
      }
      txt += "\n"
    })
    g_textarea.value(txt);
  } else if (key == 'o') {
    let txt = "[";
    temp0array.forEach((line) => {
      txt += "[";
      for (let i=0; i<line.length; i++) {
        if (i > 0) 
          txt += ",";

        txt += line[i]
      }
      txt += "],\n";
    });
    txt += "]";
    g_textarea.value(txt);
  }
}

function keyReleased() {
  if (key == 'r') {
    g_recorderviz.StopRecording();
  }
}