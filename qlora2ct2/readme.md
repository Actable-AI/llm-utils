This is some trick for faster inference qlora model using Ctranslate2 instead of normal inference.

The method basiclly just merge the adapeter of qlora model with full model, save it and then convert to ct2.

If ct2 support your base model then this method work just fine.
