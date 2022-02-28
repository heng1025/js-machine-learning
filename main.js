import * as tf from "@tensorflow/tfjs";
import Plotly from "plotly.js-dist-min";
import { trainData, testData } from "./data.js";

// markers
const dataTraceTrain = {
  x: trainData.sizeMB,
  y: trainData.timeSec,
  name: "trainData",
  mode: "markers",
  type: "scatter",
  marker: { symbol: "circle", size: 8 },
};

const dataTraceTest = {
  x: testData.sizeMB,
  y: testData.timeSec,
  name: "testData",
  mode: "markers",
  type: "scatter",
  marker: { symbol: "triangle-up", size: 10 },
};

const dataTrace10Epochs = {
  x: [0, 2],
  y: [0, 0.01],
  name: "model after N epochs",
  mode: "lines",
  line: {
    color: "blue",
    width: 1,
    dash: "dot",
  },
};

const dataTrace20Epochs = {
  ...dataTrace10Epochs,
  line: {
    color: "green",
    width: 2,
    dash: "dash",
  },
};

const dataTrace100Epochs = {
  ...dataTrace10Epochs,
  line: {
    color: "red",
    width: 3,
    dash: "longdash",
  },
};

const dataTrace200Epochs = {
  ...dataTrace10Epochs,
  line: {
    color: "black",
    width: 4,
    dash: "solid",
  },
};

// 拟合过程图
Plotly.newPlot(
  "modelFitResult",
  [
    dataTraceTrain,
    dataTraceTest,
    dataTrace10Epochs,
    dataTrace20Epochs,
    dataTrace100Epochs,
    dataTrace200Epochs,
  ],
  {
    title: "Model fit result",
    xaxis: {
      title: "size (MB)",
    },
    yaxis: {
      title: "time (sec)",
    },
  }
);

// 权重损失图
const lossSurfaceData = {
  x: [],
  y: [],
  type: "contour",
};

Plotly.newPlot("lossSurface", [lossSurfaceData], {
  title: "loss surface",
  xaxis: {
    title: "k (kernel)",
  },
  yaxis: {
    title: "b (bias)",
  },
});

const trainXs = tf.tensor2d(trainData.sizeMB, [20, 1]);
const trainYs = tf.tensor2d(trainData.timeSec, [20, 1]);

const testXs = tf.tensor2d(testData.sizeMB, [20, 1]);
const testYs = tf.tensor2d(testData.timeSec, [20, 1]);

// 创建模型
const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [1], units: 1 }));

const optimizer = tf.train.sgd("0.0005");

// 编译模型
model.compile({ optimizer, loss: "meanAbsoluteError" });

let k = 0;
let b = 0;
model.setWeights([tf.tensor2d([k], [1, 1]), tf.tensor1d([b])]);

// 训练模型
await model.fit(trainXs, trainYs, {
  epochs: 200,
  callbacks: {
    onEpochEnd: async (epochs) => {
      k = model.getWeights()[0].dataSync()[0];
      b = model.getWeights()[1].dataSync()[0];

      lossSurfaceData.x[epochs] = k;
      lossSurfaceData.y[epochs] = b;

      if (epochs === 9) {
        updateScatterWithLines(dataTrace10Epochs, k, b, 10, 2);
        updateLossSurfaceChart(lossSurfaceData);
        console.log("wrote model 10");
        console.log(`10 epochs,k=${k},b=${b}`);
      }
      if (epochs === 19) {
        updateScatterWithLines(dataTrace20Epochs, k, b, 20, 3);
        updateLossSurfaceChart(lossSurfaceData);
        console.log("wrote model 20");
        console.log(`20 epochs,k=${k},b=${b}`);
      }
      if (epochs === 99) {
        updateScatterWithLines(dataTrace100Epochs, k, b, 100, 4);
        updateLossSurfaceChart(lossSurfaceData);
        console.log("wrote model 100");
        console.log(`100 epochs,k=${k},b=${b}`);
      }
      if (epochs === 199) {
        updateScatterWithLines(dataTrace200Epochs, k, b, 200, 5);
        updateLossSurfaceChart(lossSurfaceData);
        console.log("wrote model 200");
        console.log(`200 epochs,k=${k},b=${b}`);
        console.log(`lossSurfaceData=`, lossSurfaceData);
      }
    },
  },
});

// 拟合
model.evaluate(testXs, testYs).print();

// 预测
const smallFileMB = 1;
const bigFileMB = 100;
const hugeFielMB = 10000;

model.predict(tf.tensor2d([[smallFileMB], [bigFileMB], [hugeFielMB]])).print();

function updateScatterWithLines(dataTrace, k, b, N, tranIndex) {
  dataTrace.x = [0, 10];
  dataTrace.y = [b, b + k * 10];

  const update = {
    x: [dataTrace.x],
    y: [dataTrace.y],
    name: `model after ${N} epochs`,
  };
  Plotly.restyle("modelFitResult", update, tranIndex);
}

function updateLossSurfaceChart(data, k, b) {
  const update = {
    x: [data.x],
    y: [data.y],
  };
  Plotly.restyle("lossSurface", update);
}
