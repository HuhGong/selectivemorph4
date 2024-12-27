import React, {useState} from "react";
import {FaUpload} from "react-icons/fa6";
import {FaImage} from "react-icons/fa";

function PhotoSelector() {
    const [contentImage, setContentImage] = useState(null);
    const [styleImage, setStyleImage] = useState(null);
    const [loading, setLoading] = useState(false);
    const [outputImages, setOutputImages] = useState([]); // 여러 출력 이미지 저장
    const [currentSelection, setCurrentSelection] = useState('content'); // 현재 선택된 이미지 상태 ('content' 또는 'style')
    const [selectedIds, setSelectedIds] = useState([]);
    const [finalImage, setFinalImage] = useState(null); // 최종 이미지 상태 추가


    const [contentPhotos, setContentPhotos] = useState([
        {id: 1, src: '', isUpload: true},
        {id: 2, src: require('./assets/content/image4.jpg')},
        {id: 3, src: require('./assets/content/image5.jpg')},
        {id: 4, src: require('./assets/content/image6.jpeg')},
        {id: 5, src: require('./assets/content/image8.png')},
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
    };

    const handlePhotoClick = (photo, type) => {
        const id = photo.id;

        // 갤러리에서 클릭한 경우에는 selectedIds를 변경하지 않음
        if (type === 'content' || type === 'style') {
            // 해당 이미지 클릭 시 contentImage 또는 styleImage 설정
            if (type === 'content') {
                setContentImage(photo.src);
            } else if (type === 'style') {
                setStyleImage(photo.src);
            }
            return; // ID 리스트를 변경하지 않고 종료
        }

        // 일반 클릭 처리 (outputImages 클릭 등)
        setSelectedIds(prevIds => {
            if (prevIds.includes(id)) {
                return prevIds.filter(existingId => existingId !== id);
            } else {
                return [...prevIds, id];
            }
        });
    };



    const handleFileUploadToBackend = async () => {
        if (!contentImage || !styleImage) {
            alert('Content Image와 Style Image를 모두 선택해 주세요.');
            return;
        }

        setLoading(true);
        // 업로드 전에 outputImages 배열을 비움
        setOutputImages([]);

        try {
            const response = await fetch('http://localhost:5000/upload', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    // 캐시 방지를 위한 헤더 추가
                    'Cache-Control': 'no-cache',
                    'Pragma': 'no-cache'
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
                    // // 타임스탬프를 추가하여 캐시 방지
                    // const newOutputImages = data.outputImages.map(imagePath =>
                    //     `http://localhost:5000${imagePath}?t=${new Date().getTime()}`
                    // );
                    // setOutputImages(newOutputImages);
                    const absoluteImagePaths = data.outputImages.map((imagePath, index) => ({
                        id: index, // ID 설정
                        path: `http://localhost:5000${imagePath}` // 절대 경로 설정
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
        }
    };

    const handleClassTransfer = async () => {
        setLoading(true);
        try {
            console.log('Selected IDs before sending:', selectedIds);

            const response = await fetch('http://localhost:5000/transfer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ selectedClasses: selectedIds }),
            });

            console.log('Server response:', response);  // 서버 응답 로그
            if (response.ok) {
                const data = await response.json();
                console.log('Transfer data:', data);  // 데이터 로그
                setFinalImage(`http://localhost:5000${data.finalImagePath}`); // finalImage 업데이트
                alert('클래스 트랜스퍼가 완료되었습니다!');
            } else {
                const errorData = await response.json();
                alert(`클래스 트랜스퍼 실패: ${errorData.message || response.statusText}`);
            }
        } catch (error) {
            alert('클��스 트랜스퍼 중 오류가 발생했습니다.');
            console.error('Transfer error:', error);
        } finally {
            setLoading(false);
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
        setCurrentSelection(type);
    };

    const handleGeneratedImageClick = (index) => {
        const imageObj = outputImages[index]; // 클릭한 이미지 객체 가져오기
        const imagePath = imageObj.path; // 이미지 경로 가져오기

        // 파일 경로에서 ID 추출
        const match = imagePath.match(/anno_class_img_(\d+)\.png$/);
        const id = match ? match[1] : null; // ID를 추출

        if (id !== null) {
            setSelectedIds(prevIds => {
                if (prevIds.includes(id)) {
                    return prevIds.filter(existingId => existingId !== id); // ���미 존재하면 제거
                } else {
                    return [...prevIds, id]; // 없으면 추가
                }
            });
        }
    };




    return (
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
                            disabled={loading}>
                        {loading ? '업로드 중...' : '스타일 전송을 위한 업로드'}
                    </button>
                    <div className="right-section">
                        {outputImages.length > 0 && (
                            <div className="result-container">

                            </div>
                        )}
                    </div>
                </div>
            </div>


            {/* Right Section */}
            <div className="right-section">
                {outputImages.length > 0 ? (
                    <div className="generated-image-section">
                        <h3>생성된 이미지:</h3>
                        <div className="gallery">
                            {!finalImage && outputImages.map((imageObj, index) => (
                                <div key={index} onClick={() => handleGeneratedImageClick(index)}>
                                    <img
                                        src={imageObj.path}
                                        alt={`Generated Output ${index}`}
                                        className="result-preview1"
                                    />
                                </div>
                            ))}
                            {/* 최종 이미지를 갤러리의 마지막에 표시 */}
                            {finalImage && (
                                <div className="final-image-container">
                                    <h4>최종 이미지:</h4>
                                    <img src={finalImage} alt="Final Combined" className="result-preview1" />
                                </div>
                            )}
                        </div>
                        {/* 클릭한 이미지 번호 리스트를 여기로 이동 */}
                        {selectedIds.length > 0 && (
                            <div className="image-number-display">
                                <h4>클릭한 이미지 번호:</h4>
                                <ul>
                                    {selectedIds.sort((a, b) => a - b).map(id => (
                                        <li key={id}>{id}</li>
                                    ))}
                                </ul>
                            </div>
                        )}
                        {/* 클래스를 트랜스퍼 버튼 */}
                        <button type="button" className="btn btn-primary"
                                style={{
                                    width: 'auto',
                                    height: '50px',
                                    padding: '5px 10px',
                                    fontSize: '14px',
                                    borderRadius: '0.25rem',
                                    display: 'inline-block',
                                    textAlign: 'center',
                                }}
                                onClick={handleClassTransfer}>
                            final transfer
                        </button>
                    </div>

                ) : (
                    <>
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
                                        style={{ display: 'none' }}
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
                                                    <FaUpload style={{ fontSize: '24px', color: '#888' }} />
                                                </div>
                                            ) : (
                                                <img src={photo.src} alt="Thumbnail" className="thumbnail-image" />
                                            )}
                                        </div>
                                    ))}
                                    <input
                                        id="styleUploadInputGallery"
                                        type="file"
                                        accept="image/*"
                                        style={{ display: 'none' }}
                                        onChange={(e) => handleImageUpload(e, 'style')}
                                    />
                                </div>
                            </>
                        )}
                    </>
                )}
            </div>


            {/*/!* 클래스를 트랜스퍼 버튼 *!/*/}
            {/*<button type="button" className="btn btn-primary"*/}
            {/*        style={{*/}
            {/*            width: 'auto',*/}
            {/*            height: '50px',*/}
            {/*            padding: '5px 10px',*/}
            {/*            fontSize: '14px',*/}
            {/*            borderRadius: '0.25rem',*/}
            {/*            display: 'inline-block',*/}
            {/*            textAlign: 'center',*/}
            {/*        }}*/}
            {/*        onClick={handleClassTransfer}>*/}
            {/*    final transfer*/}
            {/*</button>*/}

            {/*/!* 최종 이미지 표시 *!/*/}
            {/*{finalImage && (*/}
            {/*    <div className="final-image-display">*/}
            {/*        <h4>최종 이미지:</h4>*/}
            {/*        <img src={finalImage} alt="Final Combined" className="result-preview1" />*/}
            {/*    </div>*/}
            {/*)}*/}
        </div>
    );
}

export default PhotoSelector;
