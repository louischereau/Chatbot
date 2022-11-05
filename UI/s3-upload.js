const aws = require('aws-sdk');
const express = require('express');
const multer = require('multer');
const multerS3 = require('multer-s3');
const uuid = require('uuid').v4;
const path = require('path');
//const config = require('./config.json')
const config = require("./config.js")
const config_json = JSON.parse(JSON.stringify(config))

const app = express();


const s3Config = {
    apiVersion: '2006-03-01',
    //accessKeyId: config.accessKeyId,
    //secretAccessKey: config.secretAccessKey,
 
   }

//const s3 = new aws.S3({ apiVersion: '2006-03-01' });
// Needs AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY

const s3 = new aws.S3( s3Config );

const upload = multer({
    storage: multerS3({
        s3,
        bucket: 'individualprojectfiles',
        metadata: (req, file, cb) => {
            cb(null, { fieldName: file.fieldname });
        },
        key: (req, file, cb) => {
            //const ext = path.extname(file.originalname);
            //cb(null, `${uuid()}${ext}`);
            cb(null, file.originalname);
        }
    })
});

app.use(express.static(path.join(__dirname + '/public')))

app.post('/upload', upload.array('avatar'), (req, res) => {

    return res.json({ status: 'OK', uploaded: req.files.length });
    
});

app.listen(3001);