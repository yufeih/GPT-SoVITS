using System.Text.RegularExpressions;

namespace GptSovitsPort;

/// <summary>
/// Port of LangSegmenter from Python GPT-SoVITS
/// </summary>
public static class LangSegmenter
{
    private static readonly Dictionary<string, string> DefaultLangMap = new()
    {
        { "zh", "zh" },
        { "yue", "zh" },    // Cantonese
        { "wuu", "zh" },    // Wu Chinese
        { "zh-cn", "zh" },
        { "zh-tw", "x" },   // Traditional Chinese set to x
        { "ko", "ko" },
        { "ja", "ja" },
        { "en", "en" },
    };

    /// <summary>
    /// Check if text is full English (letters, numbers, ASCII punctuation)
    /// </summary>
    private static bool FullEn(string text)
    {
        // Pattern: must contain at least one letter, only letters/numbers/ASCII chars
        var pattern = @"^(?=.*[A-Za-z])[A-Za-z0-9\s\u0020-\u007E\u2000-\u206F\u3000-\u303F\uFF00-\uFFEF]+$";
        return Regex.IsMatch(text, pattern);
    }

    /// <summary>
    /// Extract full CJK text (Chinese/Japanese/Korean ideographs)
    /// </summary>
    private static string FullCjk(string text)
    {
        var cjkRanges = new[]
        {
            (0x4E00, 0x9FFF),        // CJK Unified Ideographs
            (0x3400, 0x4DB5),        // CJK Extension A
            (0x20000, 0x2A6DD),      // CJK Extension B
            (0x2A700, 0x2B73F),      // CJK Extension C
            (0x2B740, 0x2B81F),      // CJK Extension D
            (0x2B820, 0x2CEAF),      // CJK Extension E
            (0x2CEB0, 0x2EBEF),      // CJK Extension F
            (0x30000, 0x3134A),      // CJK Extension G
            (0x31350, 0x323AF),      // CJK Extension H
            (0x2EBF0, 0x2EE5D),      // CJK Extension H
        };

        var pattern = @"[0-9、-〜。！？.!?… /]+$";

        var cjkText = "";
        foreach (char c in text)
        {
            int codePoint = c;
            bool inCjk = cjkRanges.Any(range => codePoint >= range.Item1 && codePoint <= range.Item2);
            
            if (inCjk || Regex.IsMatch(c.ToString(), pattern))
            {
                cjkText += c;
            }
        }
        
        return cjkText;
    }

    /// <summary>
    /// Split Japanese or Korean text from mixed text
    /// </summary>
    private static List<LangSegment> SplitJaKo(string tagLang, LangSegment item)
    {
        string pattern;
        if (tagLang == "ja")
        {
            pattern = @"([\u3041-\u3096\u3099\u309A\u30A1-\u30FA\u30FC]+(?:[0-9、-〜。！？.!?… ]+[\u3041-\u3096\u3099\u309A\u30A1-\u30FA\u30FC]*)*)";
        }
        else // ko
        {
            pattern = @"([\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]+(?:[0-9、-〜。！？.!?… ]+[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]*)*)";
        }

        var langList = new List<LangSegment>();
        int tag = 0;
        
        foreach (Match match in Regex.Matches(item.Text, pattern))
        {
            if (match.Index > tag)
            {
                langList.Add(new LangSegment
                {
                    Lang = item.Lang,
                    Text = item.Text.Substring(tag, match.Index - tag)
                });
            }

            tag = match.Index + match.Length;
            langList.Add(new LangSegment
            {
                Lang = tagLang,
                Text = item.Text.Substring(match.Index, match.Length)
            });
        }

        if (tag < item.Text.Length)
        {
            langList.Add(new LangSegment
            {
                Lang = item.Lang,
                Text = item.Text.Substring(tag)
            });
        }

        return langList;
    }

    /// <summary>
    /// Merge language segments if consecutive segments have same language
    /// </summary>
    private static List<LangSegment> MergeLang(List<LangSegment> langList, LangSegment item)
    {
        if (langList.Count > 0 && item.Lang == langList[^1].Lang)
        {
            langList[^1].Text += item.Text;
        }
        else
        {
            langList.Add(item);
        }
        return langList;
    }

    /// <summary>
    /// Main method to get text segments with language tags
    /// Port of LangSegmenter.getTexts from Python
    /// </summary>
    public static List<LangSegment> GetTexts(string text, string defaultLang = "")
    {
        var langSplitter = new LangSplitter(DefaultLangMap)
        {
            MergeAcrossDigit = false
        };
        
        var substr = langSplitter.SplitByLang(text);

        var langList = new List<LangSegment>();
        bool haveNum = false;

        for (int idx = 0; idx < substr.Count; idx++)
        {
            var item = substr[idx];
            var dictItem = new LangSegment { Lang = item.Lang, Text = item.Text };

            if (dictItem.Lang == "digit")
            {
                if (!string.IsNullOrEmpty(defaultLang))
                {
                    dictItem.Lang = defaultLang;
                }
                else
                {
                    haveNum = true;
                }
                langList = MergeLang(langList, dictItem);
                continue;
            }

            // Handle short English being detected as other language
            if (FullEn(dictItem.Text))
            {
                dictItem.Lang = "en";
                langList = MergeLang(langList, dictItem);
                continue;
            }

            if (!string.IsNullOrEmpty(defaultLang))
            {
                dictItem.Lang = defaultLang;
                langList = MergeLang(langList, dictItem);
                continue;
            }
            else
            {
                // Handle non-Japanese text with Japanese characters (not including CJK)
                var jaList = new List<LangSegment>();
                if (dictItem.Lang != "ja")
                {
                    jaList = SplitJaKo("ja", dictItem);
                }

                if (jaList.Count == 0)
                {
                    jaList.Add(dictItem);
                }

                // Handle non-Korean text with Korean characters (not including CJK)
                var koList = new List<LangSegment>();
                var tempList = new List<LangSegment>();
                
                foreach (var koItem in jaList)
                {
                    if (koItem.Lang != "ko")
                    {
                        koList = SplitJaKo("ko", koItem);
                    }

                    if (koList.Count > 0)
                    {
                        tempList.AddRange(koList);
                    }
                    else
                    {
                        tempList.Add(koItem);
                    }
                    koList = new List<LangSegment>();
                }

                // No Japanese/Korean splitting occurred
                if (tempList.Count == 1)
                {
                    // Unknown language - check if it's CJK
                    if (dictItem.Lang == "x")
                    {
                        var cjkText = FullCjk(dictItem.Text);
                        if (!string.IsNullOrEmpty(cjkText))
                        {
                            dictItem = new LangSegment { Lang = "zh", Text = cjkText };
                            langList = MergeLang(langList, dictItem);
                        }
                        else
                        {
                            langList = MergeLang(langList, dictItem);
                        }
                        continue;
                    }
                    else
                    {
                        langList = MergeLang(langList, dictItem);
                        continue;
                    }
                }

                // Japanese/Korean splitting occurred
                foreach (var tempItem in tempList)
                {
                    // Unknown language - check if it's CJK
                    if (tempItem.Lang == "x")
                    {
                        var cjkText = FullCjk(tempItem.Text);
                        if (!string.IsNullOrEmpty(cjkText))
                        {
                            langList = MergeLang(langList, new LangSegment { Lang = "zh", Text = cjkText });
                        }
                        else
                        {
                            langList = MergeLang(langList, tempItem);
                        }
                    }
                    else
                    {
                        langList = MergeLang(langList, tempItem);
                    }
                }
            }
        }

        // Handle digits
        if (haveNum)
        {
            var tempList = langList;
            langList = new List<LangSegment>();
            
            for (int i = 0; i < tempList.Count; i++)
            {
                var tempItem = tempList[i];
                if (tempItem.Lang == "digit")
                {
                    if (!string.IsNullOrEmpty(defaultLang))
                    {
                        tempItem.Lang = defaultLang;
                    }
                    else if (langList.Count > 0 && i == tempList.Count - 1)
                    {
                        tempItem.Lang = langList[^1].Lang;
                    }
                    else if (langList.Count == 0 && i < tempList.Count - 1)
                    {
                        tempItem.Lang = tempList[1].Lang;
                    }
                    else if (langList.Count > 0 && i < tempList.Count - 1)
                    {
                        if (langList[^1].Lang == tempList[i + 1].Lang)
                        {
                            tempItem.Lang = langList[^1].Lang;
                        }
                        else if (langList[^1].Text.Length > 0 && 
                                 ",.!?，。！？".Contains(langList[^1].Text[^1]))
                        {
                            tempItem.Lang = tempList[i + 1].Lang;
                        }
                        else if (tempList[i + 1].Text.Length > 0 && 
                                 ",.!?，。！？".Contains(tempList[i + 1].Text[0]))
                        {
                            tempItem.Lang = langList[^1].Lang;
                        }
                        else if (tempItem.Text.Length > 0 && "。.".Contains(tempItem.Text[^1]))
                        {
                            tempItem.Lang = langList[^1].Lang;
                        }
                        else if (langList[^1].Text.Length >= tempList[i + 1].Text.Length)
                        {
                            tempItem.Lang = langList[^1].Lang;
                        }
                        else
                        {
                            tempItem.Lang = tempList[i + 1].Lang;
                        }
                    }
                    else
                    {
                        tempItem.Lang = "zh";
                    }
                }

                langList = MergeLang(langList, tempItem);
            }
        }

        // Filter 'x' (unknown)
        var finalTempList = langList;
        langList = new List<LangSegment>();
        
        for (int idx = 0; idx < finalTempList.Count; idx++)
        {
            var tempItem = finalTempList[idx];
            if (tempItem.Lang == "x")
            {
                if (langList.Count > 0)
                {
                    tempItem.Lang = langList[^1].Lang;
                }
                else if (finalTempList.Count > 1)
                {
                    tempItem.Lang = finalTempList[1].Lang;
                }
                else
                {
                    tempItem.Lang = "zh";
                }
            }

            langList = MergeLang(langList, tempItem);
        }

        return langList;
    }
}

public class LangSegment
{
    public string Lang { get; set; } = "";
    public string Text { get; set; } = "";
    
    public override bool Equals(object? obj)
    {
        if (obj is LangSegment other)
        {
            return Lang == other.Lang && Text == other.Text;
        }
        return false;
    }
    
    public override int GetHashCode()
    {
        return HashCode.Combine(Lang, Text);
    }
}
