# steam-sale-optimizer
A machine learning pipeline for optimizing game discount strategies using Steam reviews, tags, and competitor pricing. Designed for data-driven revenue maximization in the gaming industry.





```pip install -e . ```
#important links

[steam scraping](https://medium.com/@thekareemyusuf/building-a-dataset-of-steam-games-with-web-scraping-2abb02409f08)
[airflow](https://medium.com/@mrunmayee.dhapre/ml-pipeline-in-airflow-71ca7e1f03ba)
[airflow2](https://medium.com/@mohamadhasan.sarvandani/learning-apache-airflow-with-simple-examples-c1b05b4761b0)
[ I'm getting an error while importing a module from another folder.](https://www.reddit.com/r/learnpython/comments/10l5j6t/cant_import_class_from_a_module_in_another_folder/)

[another suggestion])https://stackoverflow.com/questions/73166298/cant-do-python-imports-from-another-dir)

[https://steamapi.xpaw.me/#IPlayerService/GetOwnedGames]

[Scraping information of all games from Steam with Python](https://medium.com/codex/scraping-information-of-all-games-from-steam-with-python-6e44eb01a299)

[https://www.reddit.com/r/gamedev/comments/x0qs4z/we_gathered_data_about_54000_games_in_steam_and/]

[https://www.gamedeveloper.com/business/genre-viability-on-steam-and-other-trends---an-analysis-using-review-count]

[https://medium.com/thedeephub/postgresql-integration-with-python-a-simple-guide-34b675e4bffd]
[https://db-engines.com/en/ranking]

# Notes for Myself

- I have to change my data scraping criteria. Selecting only the most recently published games may not be the best approach. I need to rethink it.
- There are many free games. From what I’ve learned, some of them use different strategies like making the game free to play but charging players for in-game items such as clothes or weapons. Others earn money through ads, especially on mobile platforms.
- Can I scrape how many copies were sold for each game ID? If I can't get this information, can I find something that is strongly related to the number of copies sold? I can try to find a dataset that includes the number of copies sold. Then, I can look for a feature that has a high correlation with it using the features I already collected.
- <img src="https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/motivation.png?raw=true" width="300">
- Reviews are in multiple languages, so we need to find a way to handle that. Also, consider how to use these reviews effectively. Should we use a multimodal model that processes both text and other features, or convert the reviews into numerical scores like sentiment polarity?
-  I will need the ```reviewer_id``` for database normalization, as I aim to include 20 reviews per game.
