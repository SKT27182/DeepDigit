import {MnistData} from './data.js';
var canvas, ctx, saveButton, clearButton;
var pos = {x:0, y:0};
var rawImage;
var model;
	
function getModel() {
	model = tf.sequential();

	model.add(tf.layers.conv2d({kernelSize: 5, filters: 6, activation: 'relu', padding: 'same', inputShape: [28, 28, 1]}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

	model.add(tf.layers.conv2d({filters: 16, kernelSize: 5, activation: 'relu', padding: 'valid'}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

	model.add(tf.layers.conv2d({filters: 120, kernelSize: 5, activation: 'relu', padding: 'valid'}));

	model.add(tf.layers.flatten());

	model.add(tf.layers.dense({units: 84, activation: 'relu'}));
	
	model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

	model.compile({optimizer: tf.train.adam(), loss: 'categoricalCrossentropy', metrics: ['accuracy']});

	return model;
}

async function train(model, data) {
	const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
	const container = { name: 'Model Training', styles: { height: '1000px' } };
	const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
  
	const BATCH_SIZE = 64;
	const TRAIN_DATA_SIZE = 5500;
	const TEST_DATA_SIZE = 1000;

	const [trainXs, trainYs] = tf.tidy(() => {
		const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
		return [
			d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
			d.labels
		];
	});

	const [testXs, testYs] = tf.tidy(() => {
		const d = data.nextTestBatch(TEST_DATA_SIZE);
		return [
			d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
			d.labels
		];
	});

	return model.fit(trainXs, trainYs, {
		batchSize: BATCH_SIZE,
		validationData: [testXs, testYs],
		epochs: 1,
		shuffle: true,
		callbacks: fitCallbacks
	});
}

function setPosition(e){
	var rect = canvas.getBoundingClientRect();
	pos.x = e.clientX-100;
	pos.y = e.clientY-100;
}

function setPositionTouch(e) {
    var rect = canvas.getBoundingClientRect();
    pos.x = e.touches[0].clientX - rect.left;
    pos.y = e.touches[0].clientY - rect.top;
}
    
function draw(e) {
	if(e.buttons!=1 && e.type != 'touchmove') return;
	ctx.beginPath();
	ctx.lineWidth = 24;
	ctx.lineCap = 'round';
	ctx.strokeStyle = 'white';
	ctx.moveTo(pos.x, pos.y);
	if (e.type == 'touchmove') {
        setPositionTouch(e);
    } else {
        setPosition(e);
    }
	ctx.lineTo(pos.x, pos.y);
	ctx.stroke();
	rawImage.src = canvas.toDataURL('image/png');
}
    
function erase() {
	ctx.fillStyle = "black";
	ctx.fillRect(0,0,280,280);
}
    
function save() {
	var raw = tf.browser.fromPixels(rawImage,1);
	var resized = tf.image.resizeBilinear(raw, [28,28]);
	var tensor = resized.expandDims(0);
    var prediction = model.predict(tensor);
    var pIndex = tf.argMax(prediction, 1).dataSync();
    
	alert(pIndex);
}
    
function init() {
	canvas = document.getElementById('canvas');
	rawImage = document.getElementById('canvasimg');
	ctx = canvas.getContext("2d");
	ctx.fillStyle = "black";
	ctx.fillRect(0,0,280,280);
	canvas.addEventListener("mousemove", draw);
	canvas.addEventListener("mousedown", setPosition);
	canvas.addEventListener("mouseenter", setPosition);
	canvas.addEventListener("touchmove", draw);
    canvas.addEventListener("touchstart", setPositionTouch);
    canvas.addEventListener("touchenter", setPositionTouch);
	saveButton = document.getElementById('sb');
	saveButton.addEventListener("click", save);
	clearButton = document.getElementById('cb');
	clearButton.addEventListener("click", erase);
}

async function run() {  
	const data = new MnistData();
	await data.load();
	const model = getModel();
	tfvis.show.modelSummary({name: 'Model Architecture'}, model);
	await train(model, data);
	init();
	alert("Training is done, try classifying your handwriting!");
}

document.addEventListener('DOMContentLoaded', run);