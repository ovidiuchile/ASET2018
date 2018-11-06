'use strict';

var express = require('express');
var app = express();
var cors = require('cors')
var server = require('http').Server(app);
var bodyParser = require('body-parser');
let multer = require('multer');

var originsWhitelist = [
    'http://localhost:4200'
];
var corsOptions = {
    origin: function(origin, callback){
          var isWhitelisted = originsWhitelist.indexOf(origin) !== -1;
          callback(null, isWhitelisted);
    },
    credentials:true
};

var storage = multer.diskStorage({
    destination: function (req, file, cb) {
      cb(null, 'uploads')
    },
    filename: function (req, file, cb) {
      cb(null, file.originalname)
    }
});
let upload = multer({ storage: storage });

server.listen(process.env.PORT || 8080, function(){
    console.log("Server connected. Listening on port: " + (process.env.PORT || 8080));
});

app.use(cors(corsOptions));

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended: true}) );

app.post('/upload', upload.any(), function(req, res) {
    res.send('File uploaded!');
    res.end();
});
