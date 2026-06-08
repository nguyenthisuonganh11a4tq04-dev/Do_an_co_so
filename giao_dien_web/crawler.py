from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

from webdriver_manager.chrome import ChromeDriverManager

import time


def crawl_facebook_comments(url):

    options = webdriver.ChromeOptions()

    options.add_argument("--disable-notifications")

    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    driver.get(url)

    time.sleep(8)

    for _ in range(5):

        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);"
        )

        time.sleep(3)

    elements = driver.find_elements(
        By.XPATH,
        '//div[@dir="auto"]'
    )

    comments = []

    for el in elements:

        text = el.text.strip()

        if len(text) > 20:

            comments.append(text)

    driver.quit()

    comments = list(set(comments))

    print("TOTAL COMMENTS:", len(comments))

    return comments[:20]