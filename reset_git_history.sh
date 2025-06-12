#!/bin/bash

# 경고 메시지 표시
echo "경고: 이 스크립트는 모든 Git 이력을 제거합니다."
echo "      이 작업은 되돌릴 수 없으며, 기존의 모든 커밋 기록이 삭제됩니다."
echo "      이 작업은 개인 프로젝트나 이력을 완전히 새로 시작하려는 경우에만 수행하세요."
echo ""
read -p "계속하시겠습니까? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "작업이 취소되었습니다."
    exit 1
fi

# Git 이력 초기화
echo "Git 이력을 초기화하는 중..."

# .git 디렉토리 삭제
rm -rf .git

# Git 저장소 초기화
git init

# 현재 모든 파일 스테이징
git add .

# 초기 커밋 생성
git commit -m "초기 커밋: 프로젝트 새로 시작"

echo ""
echo "Git 이력이 성공적으로 초기화되었습니다."
echo "이제 새로운 원격 저장소를 설정하려면 다음 명령어를 사용하세요:"
echo "  git remote add origin [새_원격_저장소_URL]"
echo "  git push -u origin main --force"
echo ""
echo "주의: --force 옵션은 원격 저장소의 내용을 덮어쓰게 됩니다."
