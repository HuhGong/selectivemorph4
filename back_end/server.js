const express = require('express');
const cors = require('cors');
const path = require('path');
const sharp = require('sharp');
const fs = require('fs');
const {exec} = require('child_process');

const app = express();
const PORT = 5000;
require('dotenv').config();
// CORS 설정
app.use(cors({
    origin: '*', // 모든 출처 허용
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Cache-Control', 'Pragma', 'Expires'],
}));

const python_interpreter = process.env.PYTHON_INTERPRETER;

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

    if (!contentImage || !styleImage) {
        return res.status(400).json({message: 'Content Image and Style Image are required.'});
    }

    console.log('Upload request received:', {contentImage, styleImage});

    const uploadsDir = path.join(__dirname, 'uploads');
    const contentImageDir = path.join(uploadsDir, 'contentImage');
    const styleImageDir = path.join(uploadsDir, 'styleImage');
    const outputImageDir = path.join(__dirname, 'output');
    const outputAnnoDir = path.join(__dirname, 'output', 'anno');

    try {
        // 디렉토리가 없으면 생성
        [contentImageDir, styleImageDir, outputImageDir, outputAnnoDir].forEach(dir => {
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, {recursive: true});
            }
        });

        // 기존 파일 삭제 함수
        const clearDirectory = (dir) => {
            if (fs.existsSync(dir)) {
                fs.readdirSync(dir).forEach(file => {
                    const filePath = path.join(dir, file);
                    fs.unlinkSync(filePath);
                });
            }
        };

        // 모든 관련 디렉토리 초기화
        clearDirectory(contentImageDir);
        clearDirectory(styleImageDir);
        clearDirectory(outputAnnoDir);

        const contentImagePath = path.join(contentImageDir, 'contentImage.png');
        const styleImagePath = path.join(styleImageDir, 'styleImage.png');

        // 이미지 처리 및 저장
        const contentImageBuffer = Buffer.from(contentImage.replace(/^data:image\/\w+;base64,/, ""), 'base64');
        const styleImageBuffer = Buffer.from(styleImage.replace(/^data:image\/\w+;base64,/, ""), 'base64');

        await sharp(contentImageBuffer).toFile(contentImagePath);
        await sharp(styleImageBuffer).toFile(styleImagePath);

        // Python 스크립트 실행
        await new Promise((resolve, reject) => {
            exec(`"${python_interpreter}" ${path.join(__dirname, 'pspnet_segment.py')} "${contentImagePath}" "${styleImagePath}"`,
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
                    // 파일 시스템 동기화를 위한 지연 추가
                    setTimeout(() => {
                        resolve();
                    }, 1000); // 1초 대기
                });
        });

        // 새로 생성된 이미지 경로만 반환
        const newOutputFiles = fs.readdirSync(outputAnnoDir)
            .filter(file => file.endsWith('.png'));
        const outputImagePaths = newOutputFiles.map(file => `/output/anno/${file}`);

        res.status(200).json({
            message: 'Images uploaded and processed successfully!',
            outputImages: outputImagePaths,
        });
    } catch (error) {
        console.error('Error processing images:', error);
        res.status(500).json({message: 'Error processing images', error: error.message});
    }
});

// 선택된 이미지 ID 처리 엔드포인트
app.post('/process-selected', (req, res) => {
    const {selectedIds} = req.body;

    if (!selectedIds || !Array.isArray(selectedIds)) {
        return res.status(400).json({message: 'Selected IDs are required and must be an array.'});
    }

    console.log('Selected IDs received:', selectedIds);

    res.status(200).json({message: 'Selected IDs processed successfully!', selectedIds});
});


app.post('/transfer', async (req, res) => {
    const {selectedClasses} = req.body;

    // selectedClasses가 빈 배열이면 transfer.py를 실행하고,
    // 그렇지 않으면 real_final_src_transfer.py를 실행
    const scriptPath = selectedClasses.length === 0
        ? path.join(__dirname, 'style_transfer_normal.py')
        : path.join(__dirname, 'psp_style_transfer.py');

    try {
        const command = selectedClasses.length === 0
            ? `"${python_interpreter}" "${scriptPath}"`
            : `"${python_interpreter}" "${scriptPath}" ${selectedClasses.join(' ')}`;

        await new Promise((resolve, reject) => {
            exec(command, {
                env: {...process.env},
                shell: true,
                encoding: 'utf-8'
            }, (error, stdout, stderr) => {
                if (error) {
                    console.error(`Error executing Python script: ${error.message}`);
                    return reject(error);
                }
                if (stderr) {
                    console.error(`Python script stderr: ${stderr}`);
                }
                console.log(`Python script output: ${stdout}`);
                resolve();
            });
        });

        const finalImagePath = path.join(__dirname, 'output', 'sample_7.png');

        if (fs.existsSync(finalImagePath)) {
            res.status(200).json({
                message: 'Transfer completed successfully!',
                finalImagePath: '/output/sample_7.png',
            });
        } else {
            res.status(404).json({message: 'Final combined image not found.'});
        }
    } catch (error) {
        console.error('Error during transfer:', error);
        res.status(500).json({message: 'Error during transfer', error: error.message});
    }
});


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