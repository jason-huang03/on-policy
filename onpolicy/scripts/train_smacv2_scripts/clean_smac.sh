ps -ef | grep -E 'StarCraftII|StarCraft2v2' | grep -v grep | cut -c 9-15 | xargs kill -9
