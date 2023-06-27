This is the trick for faster inference of qlora model using Ctranslate2 library instead of normal inference.

The method basically just merges the adapter of qlora model with the full model, saves it, and then converts it to ct2.

If ct2 supports your base model, then this method works just fine.
