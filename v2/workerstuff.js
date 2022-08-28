class MyWorkerWrapper {
  constructor() {
    this.worker = undefined; // 包含了TFJS的Web Worker
    this.model = undefined;  // 直接在主线程中执行的TFJS Model
  }

  LogMessage(msg) {
    Hello.AddLogEntry(msg);
  }

  /**
 * 尝试初始化模型，优先顺序如下
 * 1. WebWorker + WebGL
 * 2. 主线程 + WebGL
 * 4. 主线程 + CPU
 * @param {*} non_worker 是否强制不使用Web Worker
 */
  InitializeMyWorker(non_worker = false) {
    if (non_worker) {
      this.do_LoadModelNonWorker();
      return;
    }
    this.worker = new Worker("myworker.js");
    this.worker.postMessage("Hey");
    this.LogMessage("尝试在WebWorker中装入TFJS");
    // 状态转换：先是undefined，再是选用了某个后端
    this.worker.onmessage = ((event) => {
      if (event.data.TfjsVersion) {
        const be = event.data.TfjsBackend;
        console.log(be)
        switch(be) {
          case "cpu":
            this.LogMessage("似乎WebWorker只支持启用CPU后端，所以尝试在主线程中启用WebGL后端");
            setTimeout(()=>{
              this.worker.postMessage({
                "tag": "dispose",
              });
              this.worker = null;
              this.do_LoadModelNonWorker();
            }, 1000);
            break;
          case undefined:
            this.LogMessage("假定WebGL后端可用，将要预热模型。");
            this.worker.postMessage({
              "tag": "LoadModel"
            });
            break;
          case "webgl":
            this.LogMessage("WebGL后端着实可用，已启用WebGL后端。")
            break;
        }
      }
    });
  }

  async do_LoadModelNonWorker() {
    this.LogMessage("尝试在主线程中装入TFJS");
    this.model = await tf.loadLayersModel("model/model.json");
    const N = 400;
    let tb = tf.buffer([1, N, 200, 1]);
    await model.predictOnBatch(tb.toTensor());

    const be = tf.getBackend();
    if (be == "webgl") {
      this.LogMessage("在主线程中启用了WebGL");
    }
  }



  /**
   * @param {*} ffts 100个FFT频谱，所以ffts.size()必须是100
   * 注意：该函数可能异步完成
   */
  DoPrediction(ffts) {

  }
}