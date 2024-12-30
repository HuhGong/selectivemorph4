import React from 'react';
import './About.css'; // 필요에 따라 스타일 파일 추가

function About() {
    return (
        <div className="about-container">
            <h1>About Selective Morph</h1>
            <p>
                Selective Morph는 사용자가 선택한 특정 클래스에 대해 스타일 트랜스퍼를 적용하여 개인화된 이미지를 생성하는 혁신적인 플랫폼입니다.
            </p>
            <h2>주요 기능</h2>
            <ul>
                <li>사용자가 직접 사진과 스타일을 업로드할 수 있습니다.</li>
                <li>선택한 클래스에만 스타일 트랜스퍼를 적용합니다.</li>
                <li>최신 AI 기술을 활용하여 정교한 결과를 제공합니다.</li>
            </ul>
            <h2>비전</h2>
            <p>
                사용자의 창의성을 표현할 수 있는 공간을 제공하고, 다양한 예술적 스타일을 탐험하도록 돕습니다.
            </p>
            <h2>팀</h2>
            <ul>
                <li>홍길동 - 프론트엔드 개발자</li>
                <li>김철수 - 백엔드 개발자</li>
                <li>이영희 - 디자이너</li>
            </ul>
            <h2>연락처</h2>
            <p>문의사항이 있으시면 <a href="mailto:info@selectivemorph.com">info@selectivemorph.com</a>으로 연락주세요.</p>

            <h2>우리의 영웅들</h2>
            <p>AI가 없었다면 우리는 이 모든 것을 이룰 수 없었을 것입니다. 여기 우리의 영웅들, 즉 AI들이 있습니다:</p>
            <ul>
                <li>
                    <a href="[Open AI ChatGPT 사이트 주소]" target="_blank" rel="noopener noreferrer">Open AI ChatGPT</a>
                </li>
                <li>
                    <a href="[뤼튼 사이트 주소]" target="_blank" rel="noopener noreferrer">뤼튼</a>
                </li>
                <li>
                    <a href="[Perplexity 사이트 주소]" target="_blank" rel="noopener noreferrer">Perplexity</a>
                </li>
                {/* 추가 사이트를 여기에 추가 가능 */}
            </ul>

            <h2>참고 자료</h2>
            <ul>
                <li>
                    CNN 스타일 트랜스퍼 논문: <a href="https://paperswithcode.com/task/style-transfer" target="_blank" rel="noopener noreferrer">여기</a>
                </li>
                <li>
                    PSP Net 논문: <a href="https://arxiv.org/abs/1612.01105" target="_blank" rel="noopener noreferrer">여기</a>
                </li>
                <li>
                    BIS Net 논문: <a href="https://arxiv.org/abs/1808.00897" target="_blank" rel="noopener noreferrer">여기</a>
                </li>
            </ul>
        </div>
    );
}

export default About;
