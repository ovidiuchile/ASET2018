'use strict';

const express = require('express');
const app = express();
const port = 8000;
const multer = require('multer');
const bodyParser = require('body-parser');
const uuidv1 = require('uuid/v1');
const rnScriptPath = 'rn/nn_get_text.py';
const { exec } = require('child_process');
const fs = require('fs');
const request = require('request');

var storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads');
    },
    filename: function (req, file, cb) {
        cb(null, uuidv1());
    }
});
let upload = multer({ storage: storage });

app.use(express.static('public'));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended: true}));

app.post('/upload', upload.single('image'), function(req, res) {
    let filename = req.file.filename;
    // icr(__dirname + '/uploads/' + filename, req.file.originalname, function(icrResult) {
    //     res.send(icrResult);
    //     res.end();
    // });
    let filePath = __dirname + '/uploads/' + filename;
    let formData = {
        'file':{
            value: fs.readFileSync(filePath),
            options:{
                contentType: req.file.mimetype,
                filename: req.file.originalname
            }
        }
    };
    request({
      url: 'http://127.0.0.1:5000',
      method: 'POST',
      formData
    }, function(err, resp, body) {
        // res.send(body);
        console.log(body);
        res.end();
    });
});

app.listen(port, function() {
    console.log(`Example app listening on port ${port}!`);
});

let icr = function(filePath, originalname, callback) {
    exec('python ' + rnScriptPath + ' ' + filePath + ' "' + originalname + '"', function(err, stdout, stderr) {
       if (err) {
           return callback('ERR');
       };
       let rnResult = fs.readFileSync(filePath + '_icr.txt');
       return callback(rnResult);
    });
};