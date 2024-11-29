import praw
from datetime import datetime
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium_stealth import stealth
import json

class Load:

    def __init__(self, subreddit):
        self.driver = webdriver.Chrome()
        self.driver.maximize_window()
        self.seen_ids = set()
        self.subreddit = subreddit

        stealth(self.driver,
                user_agent='Chrome/116.0.5845.96',
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win64",
                webgl_vendor="Google Inc.",
                renderer="ANGLE (Intel(R) HD Graphics 630 Direct3D11 vs_5_0 ps_5_0)",
                fix_hairline=True,
                )

        self.driver.get(self.get_top_all_time(self))

    @staticmethod
    def get_top_all_time(self):
        all_time_url = f'https://www.reddit.com/r/{self.subreddit}/top/?t=all'
        return all_time_url

    def batch_posts(self, number_of_batches):
        for i in range(number_of_batches):
            self.driver.execute_script("window.scrollBy(0,5000);")
            time.sleep(1)
            self.driver.execute_script("window.scrollBy(0,5000);")
            time.sleep(2.5)



    def collect_submission_ids(self):
        full_urls = []
        articles = self.driver.find_elements(By.XPATH, '//*[@class="w-full m-0"]')

        for article in articles:
            tag = article.find_element(By.TAG_NAME, 'a')
            href = tag.get_attribute('href')
            urlString = href.__str__()

            if urlString not in self.seen_ids:
                full_urls.append(urlString)
                self.seen_ids.add(urlString)
            else:
                continue


        return full_urls

class Organize:

    def __init__(self):

        self.reddit = praw.Reddit(
            client_secret='otSwuFuI1_nNroVpPqEb6K43y04crA',
            client_id='yzo62BKProtirFS4MTFaEw',
            user_agent='MentalHealthTracker -- U of U -- by u/EmbarrassedAd4795'
        )

    def _get_date(self, utc):
        return datetime.fromtimestamp(utc).__str__()

    def _get_post_year(self, utc):
        date = self._get_date(utc)
        return int(date[:4])

    def get_submission_id(self, submission_urls):
        regex = r'https?://(?:www\.)?reddit\.com/r/[^\s/]+/comments/([a-zA-Z0-9]{6,10})'

        ids = []

        for submission in submission_urls:
            match = re.match(regex, submission)
            if match:
                post_id = match.group(1)
                ids.append(post_id)
            else:
                raise Exception('No post id found')
        return ids


    def get_top_comments(self, submission, target_limit=3):
        submission.comments.replace_more(limit=0)
        top_comments = []

        for comment in submission.comments.list():

            if comment.body in ("[deleted]", "[removed]"):
                continue

            top_comments.append({
                "comment_id":comment.id,
                "date_created":self._get_date(comment.created_utc),
                "author":comment.author.name if comment.author else None,
                "score":comment.score,
                "body":comment.body
            })

            if len(top_comments) >= target_limit:
                break

        return top_comments

    def get_submission_details(self, post_ids):
        inner_posts = []
        for post_id in post_ids:
            submission = self.reddit.submission(post_id)

            if not submission.selftext.strip():
                continue

            inner_posts.append({
                "id":submission.id,
                "title":submission.title,
                "text":submission.selftext,
                "score": submission.score,
                "num_comments": submission.num_comments,
                "top_3_comments":self.get_top_comments(submission),
                "date_created": self._get_date(submission.created_utc),
                "author": submission.author.name if submission.author else None,
                "flair": submission.link_flair_text,
                "url": submission.url,
                "nsfw": submission.over_18
            })
        return inner_posts

    def write_to_file(self, filename, data):
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)





window = Load('selfimprovement')
organizer = Organize()
all_posts = []
try:
    while len(all_posts) < 500:
        ids = window.collect_submission_ids()
        post_ids = organizer.get_submission_id(ids)
        submissions = organizer.get_submission_details(post_ids)
        all_posts.extend(submissions)
        print(len(all_posts))
        window.batch_posts(1)

except Exception as e:
    print(e)
finally:
   organizer.write_to_file('r/selfimprovement', all_posts)







