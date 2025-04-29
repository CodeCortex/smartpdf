import os
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
import time
import smtplib




def run_playwright_automation():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        page.goto("http://localhost:8501")

        pdf_file_paths = ["attension.pdf", "sql.pdf"]
        page.set_input_files("input[type='file']", pdf_file_paths)

        page.click("button:has-text('Process PDFs')")
        page.wait_for_selector("text=✅ PDFs processed and chat is ready!", timeout=60000)

        questions = [
            "What is SQL?",
            "What is covered in decoder?",
            "Explain types of joins in SQL."
        ]

        for question in questions:
            page.fill("input[type='text']", question)
            page.press("input[type='text']", "Enter")
            page.wait_for_selector("div.chat-message.bot", timeout=60000)
            time.sleep(7)

        # Download to permanent path
        with page.expect_download() as download_info:
            page.click("button:has-text('Download Chat History')")
        download = download_info.value

        permanent_path = os.path.expanduser("~/Downloads/chat_history.txt")
        download.save_as(permanent_path)
        time.sleep(5)
        print(f"Downloaded file to: {permanent_path}")

        browser.close()

        

run_playwright_automation()
