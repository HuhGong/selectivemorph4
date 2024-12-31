import React, {useState} from "react";
import {FaUpload} from "react-icons/fa6";
import {FaImage} from "react-icons/fa";
import './styles.css';

function PhotoSelector() {
    const [contentImage, setContentImage] = useState(null);
    const [styleImage, setStyleImage] = useState(null);
    const [loading, setLoading] = useState(false);
    const [outputImages, setOutputImages] = useState([]); // 여러 출력 이미지 저장
    const [currentSelection, setCurrentSelection] = useState('content'); // 현재 선택된 이미지 상태 ('content' 또는 'style')
    const [selectedIds, setSelectedIds] = useState([]);
    const [finalImage, setFinalImage] = useState(null); // 최종 이미지 상태 추가
    const [transferLoading, setTransferLoading] = useState(false);
    // 상태 추가
    const [isUploadDisabled, setIsUploadDisabled] = useState(false);
    const [isTransferDisabled, setIsTransferDisabled] = useState(false);

    const [contentPhotos, setContentPhotos] = useState([
        {id: 1, src: '', isUpload: true},
        {id: 2, src: require('./assets/content/image4.jpg')},
        {id: 3, src: require('./assets/content/image5.jpg')},
        {id: 4, src: require('./assets/content/image6.jpeg')},
        {id: 5, src: require('./assets/content/image8.jpg')},
        {id: 6, src: require('./assets/content/image11.jpg')},
    ]);

    const [stylePhotos, setStylePhotos] = useState([
        {id: 1, src: '', isUpload: true},
        {id: 2, src: require('./assets/style/image1.jpg')},
        {id: 3, src: require('./assets/style/image2.jpg')},
        {id: 4, src: require('./assets/style/image3.jpg')},
        {id: 5, src: require('./assets/style/image4.jpg')},
        {id: 6, src: require('./assets/style/image6.jpg')},
    ]);

    const handleImageUpload = (e, type) => {
        if (e.target.files && e.target.files[0]) {
            const reader = new FileReader();
            reader.onload = (event) => {
                if (type === 'content') {
                    setContentImage(event.target.result);
                } else if (type === 'style') {
                    setStyleImage(event.target.result);
                }
            };
            reader.readAsDataURL(e.target.files[0]);
        }
        // Reset the file input value
        e.target.value = null;
    };

    const handlePhotoClick = (photo, type) => {
        const id = photo.id;

        if (photo.isUpload) {
            document.getElementById(`${type}UploadInput`).click();
            setSelectedIds([]); // 업로드 버튼 클릭 시 초기화
            return;
        }

        // content나 style 갤러리 이미지 클릭 시
        if (type === 'content' || type === 'style') {
            if (type === 'content') {
                setContentImage(photo.src);
            } else if (type === 'style') {
                setStyleImage(photo.src);
            }
            setSelectedIds([]); // 갤러리 이미지 클릭 시 초기화
            return;
        }

        // 생성된 이미지 클릭 시
        setSelectedIds(prevIds => {
            if (prevIds.includes(id)) {
                return prevIds.filter(existingId => existingId !== id);
            } else {
                return [...prevIds, id];
            }
        });
    }


    const handleFileUploadToBackend = async () => {
        if (!contentImage || !styleImage) {
            alert('Content Image와 Style Image를 모두 선택해 주세요.');
            return;
        }

        setLoading(true);
        // 업로드 전에 outputImages 배열을 비움
        setOutputImages([]);
        setIsTransferDisabled(true); // 트랜스퍼 버튼 비활성화
        try {
            const response = await fetch('http://localhost:5000/upload', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                },
                body: JSON.stringify({
                    contentImage: contentImage.startsWith('data:image/') ? contentImage : await convertToBase64(contentImage),
                    styleImage: styleImage.startsWith('data:image/') ? styleImage : await convertToBase64(styleImage),
                }),
            });

            if (response.ok) {
                const data = await response.json();
                alert('이미지 업로드가 완료되었습니다!');

                if (data.outputImages) {
                    const absoluteImagePaths = data.outputImages.map((imagePath, index) => ({
                        id: index, // ID 설정
                        // 타임스탬프를 추가하여 캐시 방지
                        path: `http://localhost:5000${imagePath}?t=${new Date().getTime()}`
                    }));
                    setOutputImages(absoluteImagePaths); // 상태 업데이트
                } else {
                    console.error('Output images are undefined.');
                    alert('서버에서 출력 이미지를 받지 못했습니다.');
                }
            } else {
                const errorData = await response.json();
                alert(`이미지 업로드 실패: ${errorData.message || response.statusText}`);
                console.error('Upload error:', errorData);
            }
        } catch (error) {
            alert('이미지 업로드 중 오류가 발생했습니다.');
            console.error('Fetch error:', error);
        } finally {
            setLoading(false);
            setIsTransferDisabled(false); // 트랜스퍼 버튼 다시 활성화
        }
    };

    // 먼저 상단에 class_names 객체를 추가
    const class_names = {
        0: "Background", 1: "Aeroplane", 2: "Bicycle", 3: "Bird", 4: "Boat", 5: "Bottle", 6: "Bus", 7: "Car",
        8: "Cat", 9: "Chair", 10: "Cow", 11: "Dining Table", 12: "Dog", 13: "Horse", 14: "Motorbike",
        15: "Person", 16: "Potted Plant", 17: "Sheep", 18: "Sofa", 19: "Train", 20: "TV Monitor"
    };

    const handleClassTransfer = async () => {
        setTransferLoading(true);
        setIsUploadDisabled(true); // 업로드 버튼 비활성화
        try {
            console.log('Selected IDs before sending:', selectedIds);
            const response = await fetch('http://localhost:5000/transfer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    selectedClasses: selectedIds,
                }),
            });

            if (response.ok) {
                const data = await response.json();
                // 타임스탬프를 추가하여 강제로 새로운 이미지 로드
                const timestamp = new Date().getTime();
                setFinalImage(`http://localhost:5000${data.finalImagePath}?t=${timestamp}`);
                // 선택된 클래스 초기화
                setSelectedIds([]);
                alert('클래스 트랜스퍼가 완료되었습니다!');
            } else {
                const errorData = await response.json();
                alert(`클래스 트랜스퍼 실패: ${errorData.message || response.statusText}`);
            }
        } catch (error) {
            alert('클래스 트랜스퍼 중 오류가 발생했습니다.');
            console.error('Transfer error:', error);
        } finally {
            setTransferLoading(false);
            setIsUploadDisabled(false); // 업로드 버튼 다시 활성화
        }
    };


    const convertToBase64 = (url) => {
        return new Promise((resolve, reject) => {
            fetch(url)
                .then((res) => res.blob())
                .then((blob) => {
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result);
                    reader.onerror = (err) => reject(err);
                    reader.readAsDataURL(blob);
                })
                .catch((err) => reject(err));
        });
    };

    const toggleSelection = (type) => {
        // 현재 선택된 타입과 같은 타입을 클릭하면 선택 해제
        if (currentSelection === type) {
            setCurrentSelection('');
        } else {
            setCurrentSelection(type);
        }
    };


    const handleGeneratedImageClick = (index) => {
        const imageObj = outputImages[index];
        const match = imageObj.path.match(/anno_class_img_(\d+)\.png/);

        if (match) {
            const id = parseInt(match[1]);
            setSelectedIds(prevIds => {
                // 이미 선택된 ID인 경우 해당 ID를 제외한 배열 반환
                if (prevIds.includes(id)) {
                    return prevIds.filter(existingId => existingId !== id);
                }
                // 선택되지 않은 경우 새로운 ID 추가
                return [...prevIds, id];
            });

            // 이미지 요소에 selected 클래스 토글
            const imageElement = document.querySelector(`[data-index="${index}"]`);
            if (imageElement) {
                imageElement.classList.toggle('selected');
            }
        }
    };



    return (
        <div className="style-transfer-container-1">
            <div className="container">
                {/* Left Section */}
                <div className="left-section">
                    <div className="image-box">
                        <h3>Content Image</h3>
                        <div
                            className={`image-preview-container ${
                                currentSelection === 'content' ? 'selected' : ''
                            }`}
                            onClick={() => toggleSelection('content')}
                        >
                            {contentImage ? (
                                <img src={contentImage} alt="Content" className="image-preview"/>
                            ) : (
                                <div className="placeholder">
                                    <FaImage style={{fontSize: '24px', color: '#888'}}/>
                                </div>
                            )}
                        </div>
                        <input
                            type="file"
                            accept="image/*"
                            style={{display: 'none'}}
                            id="contentUploadInput"
                            onChange={(e) => handleImageUpload(e, 'content')}
                        />
                    </div>

                    <div className="image-box">
                        <h3>Style Image</h3>
                        <div
                            className={`image-preview-container ${
                                currentSelection === 'style' ? 'selected' : ''
                            }`}
                            onClick={() => toggleSelection('style')}
                        >
                            {styleImage ? (
                                <img src={styleImage} alt="Style" className="image-preview"/>
                            ) : (
                                <div className="placeholder">
                                    <FaImage style={{fontSize: '24px', color: '#888'}}/>
                                </div>
                            )}
                        </div>
                        <input
                            type="file"
                            accept="image/*"
                            style={{display: 'none'}}
                            id="styleUploadInput"
                            onChange={(e) => handleImageUpload(e, 'style')}
                        />
                        {/* Upload Button */}
                        <button type="button" className="btn btn-primary" onClick={handleFileUploadToBackend}
                                disabled={loading || isUploadDisabled}>
                            {loading ? 'Uploading...' : 'Image Upload'}
                        </button>

                        {/* 클래스를 트랜스퍼 버튼 - outputImages가 있을 때만 표시 */}
                        {outputImages.length > 0 && (
                            <button
                                className="btn btn-primary"
                                onClick={handleClassTransfer}
                                disabled={transferLoading || isTransferDisabled}
                            >
                                {transferLoading ? 'Processing...' : 'Style Transfer'}
                            </button>
                        )}
                    </div>
                </div>


                {/* Right Section */}
                <div className="right-section">
                    {currentSelection === 'content' && (
                        <>
                            <h3>Content Gallery</h3>
                            <div className="gallery">
                                {contentPhotos.map((photo) => (
                                    <div
                                        key={photo.id}
                                        onClick={() => handlePhotoClick(photo, 'content')}
                                        className={`thumbnail ${selectedIds.includes(photo.id) ? 'selected' : ''}`}
                                    >
                                        {photo.isUpload ? (
                                            <div className="thumbnail">
                                                <FaUpload style={{fontSize: '24px', color: '#888'}}/>
                                            </div>
                                        ) : (
                                            <img src={photo.src} alt="Thumbnail" className="thumbnail-image"/>
                                        )}
                                    </div>
                                ))}
                                <input
                                    id="contentUploadInputGallery"
                                    type="file"
                                    accept="image/*"
                                    style={{display: 'none'}}
                                    onChange={(e) => handleImageUpload(e, 'content')}
                                />
                            </div>
                        </>
                    )}

                    {currentSelection === 'style' && (
                        <>
                            <h3>Style Gallery</h3>
                            <div className="gallery">
                                {stylePhotos.map((photo) => (
                                    <div
                                        key={photo.id}
                                        onClick={() => handlePhotoClick(photo, 'style')}
                                        className={`thumbnail ${selectedIds.includes(photo.id) ? 'selected' : ''}`}
                                    >
                                        {photo.isUpload ? (
                                            <div className="thumbnail">
                                                <FaUpload style={{fontSize: '24px', color: '#888'}}/>
                                            </div>
                                        ) : (
                                            <img src={photo.src} alt="Thumbnail" className="thumbnail-image"/>
                                        )}
                                    </div>
                                ))}
                                <input
                                    id="styleUploadInputGallery"
                                    type="file"
                                    accept="image/*"
                                    style={{display: 'none'}}
                                    onChange={(e) => handleImageUpload(e, 'style')}
                                />
                            </div>
                        </>
                    )}
                </div>

                {selectedIds.length > 0 && (
                    <div className="image-number-display">
                        <h4>Clicked</h4>
                        <ul>
                            {selectedIds
                                .sort((a, b) => a - b)
                                .map(id => (
                                    <li key={id}>{class_names[id]}</li>
                                ))}
                        </ul>
                    </div>
                )}


                {/* 최종 이미지 표시 */}
                {finalImage && (
                    <div className="final-image-display">
                        <h4>Generated Image</h4>
                        <img src={finalImage} alt="Final Combined" className="result-preview1"/>
                    </div>
                )}
            </div>

            {outputImages.length > 0 && (
                <div className="result-container">
                    <h4>Segmented Image
                    </h4>
                    <div className="generated-images-wrapper">
                        {outputImages.map((imageObj, index) => (
                            <div key={index} onClick={() => handleGeneratedImageClick(index)}>
                                <img
                                    src={imageObj.path}
                                    alt={`Generated Output ${index}`}
                                    className="result-preview1"
                                    data-index={index}
                                />
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

export default PhotoSelector;
