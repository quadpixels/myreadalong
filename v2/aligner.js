class Aligner {
  constructor() {
    this.lang_idx = 0; // 0:简体 1:繁体
    this.LoadSampleData();
  }

  /**
   * 
   * @param {*} data 
   * @param {*} title 
   */
  LoadData(data, title) {
    this.timestamp0 = millis();
    this.timestamp1 = millis();
    this.activation_lidx = -1;
    this.title = title;
    this.data = data.slice();
  }

  view(v) {
    if (this.data == undefined) return;
    let d = this.data;
    let c = [];
    for (let i=0; i<d.length; i++) {
      c.push(m('div'), d[i][this.lang_idx].split("").map((ch) => {
        return m('span', {style:"border:1px #CCC dashed; padding:2px"}, ch);
      }));
    }
    return m('div', "(Aligner status)", c);
  }

  LoadSampleData() {
    console.log("LoadSampleData")
    this.LoadData([
      ["海上生明月", "海上生明月", "hai shang sheng ming yue"],
      ["天涯共此时", "天涯共此時", "tian ya gong ci shi"]
    ],
    "望月怀远"
    );
  }
}