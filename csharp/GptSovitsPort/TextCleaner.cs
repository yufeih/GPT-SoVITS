namespace GptSovitsPort;

/// <summary>
/// Text cleaning and phoneme conversion functions.
/// Ports Python's clean_text_inf function.
/// </summary>
public static class TextCleaner
{
    /// <summary>
    /// Cleans text and converts to phonemes.
    /// Port of Python's clean_text_inf function.
    /// </summary>
    /// <param name="text">Input text to clean</param>
    /// <param name="language">Language code (zh, en, ja, ko, yue)</param>
    /// <param name="version">Model version (v1 or v2)</param>
    /// <returns>Tuple of (phones, word2ph, norm_text)</returns>
    public static (List<string> phones, List<int>? word2ph, string normText) CleanTextInf(
        string text, string language, string version)
    {
        language = language.Replace("all_", "");
        var (phones, word2ph, normText) = CleanText(text, language, version);
        var phoneIds = CleanedTextToSequence(phones, version);
        return (phoneIds, word2ph, normText);
    }
    
    /// <summary>
    /// Core text cleaning logic - stub implementation.
    /// In real implementation, this would call language-specific g2p (grapheme-to-phoneme) converters.
    /// </summary>
    private static (List<string> phones, List<int>? word2ph, string normText) CleanText(
        string text, string language, string version)
    {
        // This is a STUB - for testing, return one "phone" per character
        // Real implementation would:
        // 1. Import language-specific module (chinese2, japanese, english, korean, cantonese)
        // 2. Call text_normalize if available
        // 3. Call g2p (grapheme to phoneme) conversion
        // 4. For zh/yue, return word2ph; for others, return null
        
        var phones = new List<string>();
        for (int i = 0; i < text.Length; i++)
        {
            phones.Add(i.ToString()); // Mock: use index as phone ID
        }
        
        List<int>? word2ph = language == "zh" || language == "yue" 
            ? text.Select(_ => 1).ToList() 
            : null;
        
        return (phones, word2ph, text);
    }
    
    /// <summary>
    /// Converts cleaned phoneme text to sequence of IDs.
    /// Port of Python's cleaned_text_to_sequence function.
    /// </summary>
    public static List<string> CleanedTextToSequence(List<string> cleanedText, string version)
    {
        // In real implementation, this would map phoneme symbols to integer IDs
        // using symbols_v1 or symbols_v2 dictionaries
        // For now, return the phonemes as-is for testing
        return cleanedText;
    }
}
