from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

import time
import urllib.request

class WebCrawling:
    def __init__(self, address, search_value):
        self.__driver = webdriver.Chrome(ChromeDriverManager().install())
        self.__address = address
        self.__search_value = search_value
        self.__elements_dict = {
            "id": By.ID,
            "link_text": By.LINK_TEXT,
            "partital_link_text": By.PARTIAL_LINK_TEXT,
            "name": By.NAME,
            "class_name": By.CLASS_NAME,
            "css_selector": By.CSS_SELECTOR,
            "tag_name": By.TAG_NAME,
            "xpath": By.XPATH
        }
        
    def get_address(self, address):
        self.__driver.get(address)
        
    def __get_element(self, tag, search_value):
        element = self.__driver.find_element(self.__elements_dict[tag], search_value)
        return element
    
    def __get_elements(self, tag, search_value):
        elements = self.__driver.find_elements(self.__elements_dict[tag], search_value)
        return elements
        
    def search(self, search_value):
        elem = self.__get_element("name", "q")
        elem.send_keys(search_value)
        elem.send_keys(Keys.RETURN)
        
    def scrolling(self):
        elem = self.__get_element("css_selector", ".mye4qd")
        last_height = self.__driver.execute_script("return document.body.scrollHeight")
        
        while True:
            self.__driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            time.sleep(1)
            
            new_height = self.__driver.execute_script("return document.body.scrollHeight")
            
            if new_height == last_height:
                try:
                    elem.click()
                except:
                    break
            
            last_height = new_height
            
    def get_images(self):
        images = self.__get_elements("css_selector", ".rg_i.Q4LuWd")
        
        for img in images:
            try:
                img.click()
                time.sleep(3)
                
                img_url = self.__get_element("css_selector", ".n3VNCb").get_attribute("src")
                opener = urllib.request.build_opener()
                opener.addheaders = [('Users-Agents', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
                
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(img_url, "_.jpg")
            except:
                pass
            
        self.__driver.close()
    
    def quit_browser(self):
        self.__driver.quit()
        
    def main(self):
        self.get_address(self.__address)
        self.search(self.__search_value)
        self.scrolling()
        self.get_images()
        
        
        
if __name__ == "__main__":
    run = WebCrawling("https://www.google.co.kr/imghp?=hlko&ogbl", "test")
    run.main()
    for i in range(10, 0, -1):
        print(f"{i}초 후 브라우저가 종료 됩니다.")
    run.quit_browser()