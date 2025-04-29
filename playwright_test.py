from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        query = "apple"
        page.goto(f"https://duckduckgo.com/?q={query}")

        page.wait_for_timeout(3000)
        print(page.title())

        input("Press Enter to close the browser...")  
        browser.close()

if __name__ == "__main__":
    run()
