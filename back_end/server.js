const express = require('express');
const cors = require('cors');
const path = require('path');
const sharp = require('sharp');
const fs = require('fs');
const {exec} = require('child_process');

const app = express();
const PORT = 5000;

// CORS 설정
app.use(cors({
    origin: '*', // 모든 출처 허용
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type'],
}));

const python_interpreter = "C:\\Users\\edu\\anaconda3\\envs\\kaka\\python.exe";

// 요청 본문 크기 제한 설정 (30MB로 설정)
app.use(express.json({limit: '30mb'}));
app.use(express.urlencoded({limit: '30mb', extended: true}));

// 정적 파일 제공
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
app.use('/output', express.static(path.join(__dirname, 'output')));
// Add this line to serve static files from output/anno
app.use('/output/anno', express.static(path.join(__dirname, 'output', 'anno')));

// 이미지 업로드 엔드포인트
app.post('/upload', async (req, res) => {
    const {contentImage, styleImage} = req.body;

    // 요청 데이터 유효성 검사
    if (!contentImage || !styleImage) {
        return res.status(400).json({message: 'Content Image and Style Image are required.'});
    }

    console.log('Upload request received:', {contentImage, styleImage});

    const uploadsDir = path.join(__dirname, 'uploads');
    const contentImageDir = path.join(uploadsDir, 'contentImage');
    const styleImageDir = path.join(uploadsDir, 'styleImage');
    const outputImageDir = path.join(__dirname, 'output');

    // 필요한 디렉토리 생성
    fs.mkdirSync(contentImageDir, {recursive: true});
    fs.mkdirSync(styleImageDir, {recursive: true});
    fs.mkdirSync(outputImageDir, {recursive: true});

    const contentImagePath = path.join(contentImageDir, 'contentImage.png');
    const styleImagePath = path.join(styleImageDir, 'styleImage.png');

    try {
        // Base64 데이터를 Buffer로 변환
        const contentImageBuffer = Buffer.from(contentImage.replace(/^data:image\/\w+;base64,/, ""), 'base64');
        const styleImageBuffer = Buffer.from(styleImage.replace(/^data:image\/\w+;base64,/, ""), 'base64');

        // 이미지 저장
        await sharp(contentImageBuffer).toFile(contentImagePath);
        await sharp(styleImageBuffer).toFile(styleImagePath);

        // Python 스크립트 실행
        await new Promise((resolve, reject) => {
            exec(`"${python_interpreter}"  ${path.join(__dirname, 'segment.py')} "${contentImagePath}" "${styleImagePath}"`,
                {
                    env: {...process.env},
                    shell: true
                },
                (error, stdout, stderr) => {
                    if (error) {
                        console.error(`Error executing Python script: ${error.message}`);
                        return reject(error);
                    }
                    if (stderr) {
                        console.error(`Python script stderr: ${stderr}`);
                    }
                    resolve();
                });
        });

        // 변환된 이미지 경로를 Base64 형식으로 반환
        const outputAnnoDir = path.join(__dirname, 'output', 'anno');
        const outputFiles = fs.readdirSync(outputAnnoDir)
            .filter(file => file.endsWith('.png'));
        const outputImagePaths = outputFiles.map(file => `/output/anno/${file}`);

        res.status(200).json({
            message: 'Images uploaded and processed successfully!',
            outputImages: outputImagePaths, // PNG 파일 경로 반환
        });
    } catch (error) {
        console.error('Error processing images:', error);
        res.status(500).json({message: 'Error processing images', error: error.message});
    }
});

// 서버 코드는 그대로 유지
app.get('/annotations', (req, res) => {
    const annotationsDir = path.join(__dirname, 'output');

    fs.readdir(annotationsDir, (err, files) => {
        if (err) {
            console.error('Error reading annotations directory:', err);
            return res.status(500).json({message: 'Error reading annotations directory', error: err.message});
        }

        const annotationFiles = files
            .filter(file => {
                const filePath = path.join(annotationsDir, file);
                return fs.statSync(filePath).isFile() &&
                    file.startsWith('anno/') &&
                    file.endsWith('.png');
            });

        const annotationPaths = annotationFiles.map(file => path.join('/output', file));

        res.status(200).json({
            message: 'Annotation images retrieved successfully!',
            annotations: annotationPaths,
        });
    });
});


// 기본 오류 처리 미들웨어
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({message: 'Internal Server Error'});
});

// 서버 시작
app.listen(PORT, '0.0.0.0', () => {
    console.log(`서버가 ${PORT}에서 실행 중입니다.`);
});
