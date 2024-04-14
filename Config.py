class Config():
    def __init__(self, path, valid_fliter=True, weight="distance", number=-1, tqdm=False) -> None:
        self.content = {"DataFilePath": path,
                        "ValidFliter": valid_fliter,
                        "Weight": weight,
                        "Number": number,
                        "Tqdm" : tqdm}
    
    def __getitem__(self, key):
        return self.content[key]