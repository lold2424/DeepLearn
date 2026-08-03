# -*- coding: utf-8 -*-
"""
Pexels / Pixabay 공식 API로 이미지를 검색해서 다운로드하는 스크립트.
(구글 이미지 직접 크롤링은 저작권/이용약관 문제가 있어 사용하지 않음)

사전 준비:
    pip install requests

API 키 발급 (둘 다 무료, 가입만 하면 즉시 발급):
    Pexels:  https://www.pexels.com/api/         (Get Started -> API Key)
    Pixabay: https://pixabay.com/api/docs/        (가입 후 계정 페이지에서 확인)

사용법:
    python download_stock_images.py --source pexels  --query "empty food tray plastic" --api_key YOUR_KEY --out_dir ./table_images --max_images 30
    python download_stock_images.py --source pixabay --query "disposable tray"          --api_key YOUR_KEY --out_dir ./table_images --max_images 30
"""

import os
import argparse
import requests
import time


def download_pexels(query, api_key, out_dir, max_images):
    headers = {"Authorization": api_key}
    per_page = 80
    downloaded = 0
    page = 1

    while downloaded < max_images:
        resp = requests.get(
            "https://api.pexels.com/v1/search",
            headers=headers,
            params={"query": query, "per_page": per_page, "page": page},
        )
        if resp.status_code != 200:
            print(f"Pexels API 오류: {resp.status_code} {resp.text}")
            break

        data = resp.json()
        photos = data.get("photos", [])
        if not photos:
            print("더 이상 결과 없음")
            break

        for photo in photos:
            if downloaded >= max_images:
                break
            img_url = photo["src"]["large"]  # 필요시 'original'로 변경 가능
            img_id = photo["id"]
            save_path = os.path.join(out_dir, f"pexels_{img_id}.jpg")

            img_resp = requests.get(img_url)
            with open(save_path, "wb") as f:
                f.write(img_resp.content)

            downloaded += 1
            print(f"[{downloaded}/{max_images}] 저장됨: {save_path}")
            time.sleep(0.3)  # API rate limit 배려

        page += 1

    print(f"\nPexels에서 총 {downloaded}장 다운로드 완료")


def download_pixabay(query, api_key, out_dir, max_images):
    downloaded = 0
    page = 1
    per_page = 50

    while downloaded < max_images:
        resp = requests.get(
            "https://pixabay.com/api/",
            params={
                "key": api_key,
                "q": query,
                "image_type": "photo",
                "per_page": per_page,
                "page": page,
                "safesearch": "true",
            },
        )
        if resp.status_code != 200:
            print(f"Pixabay API 오류: {resp.status_code} {resp.text}")
            break

        data = resp.json()
        hits = data.get("hits", [])
        if not hits:
            print("더 이상 결과 없음")
            break

        for hit in hits:
            if downloaded >= max_images:
                break
            img_url = hit["largeImageURL"]
            img_id = hit["id"]
            save_path = os.path.join(out_dir, f"pixabay_{img_id}.jpg")

            img_resp = requests.get(img_url)
            with open(save_path, "wb") as f:
                f.write(img_resp.content)

            downloaded += 1
            print(f"[{downloaded}/{max_images}] 저장됨: {save_path}")
            time.sleep(0.3)

        page += 1

    print(f"\nPixabay에서 총 {downloaded}장 다운로드 완료")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["pexels", "pixabay"], required=True)
    ap.add_argument("--query", required=True, help='검색 키워드 (예: "empty food tray plastic")')
    ap.add_argument("--api_key", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_images", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.source == "pexels":
        download_pexels(args.query, args.api_key, args.out_dir, args.max_images)
    else:
        download_pixabay(args.query, args.api_key, args.out_dir, args.max_images)


if __name__ == "__main__":
    main()
