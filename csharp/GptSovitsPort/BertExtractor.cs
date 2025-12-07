namespace GptSovitsPort;

/// <summary>
/// BERT feature extraction functions.
/// Port of Python's get_bert_inf function.
/// </summary>
public static class BertExtractor
{
    /// <summary>
    /// Gets BERT features for phonemes.
    /// Port of Python's get_bert_inf function.
    /// </summary>
    /// <param name="phones">List of phoneme strings</param>
    /// <param name="word2ph">Word to phoneme mapping (can be null)</param>
    /// <param name="normText">Normalized text</param>
    /// <param name="language">Language code</param>
    /// <param name="isHalf">Whether to use half precision (float16)</param>
    /// <returns>BERT feature tensor as 2D array [1024, phone_count]</returns>
    public static float[][] GetBertInf(
        List<string> phones, 
        List<int>? word2ph, 
        string normText, 
        string language,
        bool isHalf = true)
    {
        language = language.Replace("all_", "");
        
        if (language == "zh")
        {
            // In real implementation, this would:
            // 1. Use BERT tokenizer to encode normText
            // 2. Run BERT model to get hidden states
            // 3. Concatenate last 3-2 hidden layers
            // 4. Repeat features according to word2ph mapping
            
            // For now, return zeros as placeholder
            return CreateZeroTensor(1024, phones.Count);
        }
        else
        {
            // For non-Chinese, return zero tensor
            return CreateZeroTensor(1024, phones.Count);
        }
    }
    
    private static float[][] CreateZeroTensor(int dim1, int dim2)
    {
        var tensor = new float[dim1][];
        for (int i = 0; i < dim1; i++)
        {
            tensor[i] = new float[dim2];
        }
        return tensor;
    }
}
