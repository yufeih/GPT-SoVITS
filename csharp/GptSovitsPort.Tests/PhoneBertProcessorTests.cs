using Xunit;
using GptSovitsPort;
using System.Text.Json;

namespace GptSovitsPort.Tests;

/// <summary>
/// Unit tests for PhoneBertProcessor.GetPhonesAndBert method.
/// These tests verify the C# port matches the Python implementation behavior.
/// </summary>
public class PhoneBertProcessorTests
{
    private readonly PhoneBertProcessor _processor;

    public PhoneBertProcessorTests()
    {
        _processor = new PhoneBertProcessor(isHalf: false);
    }

    [Fact]
    public void TestSimpleChinese()
    {
        // Arrange
        var text = "你好世界";
        var language = "all_zh";
        var version = "v2";

        // Act
        var (phones, bert, normText) = _processor.GetPhonesAndBert(text, language, version);

        // Assert
        Assert.NotNull(phones);
        Assert.NotNull(bert);
        Assert.NotNull(normText);
        Assert.Equal(1024, bert.Length); // BERT dimension

        SaveTestResult("simple_chinese", text, language, version, phones, bert, normText);
    }

    [Fact]
    public void TestSimpleEnglish()
    {
        // Arrange
        var text = "Hello world";
        var language = "en";
        var version = "v2";

        // Act
        var (phones, bert, normText) = _processor.GetPhonesAndBert(text, language, version);

        // Assert
        Assert.NotNull(phones);
        Assert.NotNull(bert);
        Assert.NotNull(normText);
        Assert.Equal(1024, bert.Length);

        SaveTestResult("simple_english", text, language, version, phones, bert, normText);
    }

    [Fact]
    public void TestMixedLanguage()
    {
        // Arrange
        var text = "你好Hello世界World";
        var language = "zh"; // Mixed mode
        var version = "v2";

        // Act
        var (phones, bert, normText) = _processor.GetPhonesAndBert(text, language, version);

        // Assert
        Assert.NotNull(phones);
        Assert.NotNull(bert);
        Assert.Equal(1024, bert.Length);

        SaveTestResult("mixed_language", text, language, version, phones, bert, normText);
    }

    [Fact]
    public void TestAutoLanguageDetection()
    {
        // Arrange
        var text = "HelloWorld";
        var language = "auto";
        var version = "v2";

        // Act
        var (phones, bert, normText) = _processor.GetPhonesAndBert(text, language, version);

        // Assert
        Assert.NotNull(phones);
        Assert.NotNull(bert);
        Assert.Equal(1024, bert.Length);

        SaveTestResult("auto_detection", text, language, version, phones, bert, normText);
    }

    [Fact]
    public void TestShortTextPadding()
    {
        // Arrange
        var text = "Hi";
        var language = "en";
        var version = "v2";

        // Act
        var (phones, bert, normText) = _processor.GetPhonesAndBert(text, language, version);

        // Assert - Short text should trigger recursive call with "." prefix
        Assert.NotNull(phones);
        Assert.NotNull(bert);

        SaveTestResult("short_text", text, language, version, phones, bert, normText);
    }

    [Fact]
    public void TestMultipleSpaces()
    {
        // Arrange
        var text = "Hello    World";
        var language = "en";
        var version = "v2";

        // Act
        var (phones, bert, normText) = _processor.GetPhonesAndBert(text, language, version);

        // Assert
        Assert.NotNull(phones);
        Assert.NotNull(bert);

        SaveTestResult("multiple_spaces", text, language, version, phones, bert, normText);
    }

    [Fact]
    public void TestJapanese()
    {
        // Arrange
        var text = "こんにちは";
        var language = "all_ja";
        var version = "v2";

        // Act
        var (phones, bert, normText) = _processor.GetPhonesAndBert(text, language, version);

        // Assert
        Assert.NotNull(phones);
        Assert.NotNull(bert);
        Assert.Equal(1024, bert.Length);

        SaveTestResult("japanese", text, language, version, phones, bert, normText);
    }

    [Fact]
    public void TestKorean()
    {
        // Arrange
        var text = "안녕하세요";
        var language = "all_ko";
        var version = "v2";

        // Act
        var (phones, bert, normText) = _processor.GetPhonesAndBert(text, language, version);

        // Assert
        Assert.NotNull(phones);
        Assert.NotNull(bert);
        Assert.Equal(1024, bert.Length);

        SaveTestResult("korean", text, language, version, phones, bert, normText);
    }

    [Fact]
    public void TestCantonese()
    {
        // Arrange
        var text = "你好";
        var language = "all_yue";
        var version = "v2";

        // Act
        var (phones, bert, normText) = _processor.GetPhonesAndBert(text, language, version);

        // Assert
        Assert.NotNull(phones);
        Assert.NotNull(bert);
        Assert.Equal(1024, bert.Length);

        SaveTestResult("cantonese", text, language, version, phones, bert, normText);
    }

    [Fact]
    public void TestAutoYue()
    {
        // Arrange
        var text = "你好World";
        var language = "auto_yue";
        var version = "v2";

        // Act
        var (phones, bert, normText) = _processor.GetPhonesAndBert(text, language, version);

        // Assert
        Assert.NotNull(phones);
        Assert.NotNull(bert);
        Assert.Equal(1024, bert.Length);

        SaveTestResult("auto_yue", text, language, version, phones, bert, normText);
    }

    /// <summary>
    /// Saves test results to JSON for cross-language validation with Python.
    /// </summary>
    private void SaveTestResult(string testName, string text, string language, string version,
        List<string> phones, float[][] bert, string normText)
    {
        var result = new
        {
            test_name = testName,
            input = new
            {
                text,
                language,
                version
            },
            output = new
            {
                phones,
                phones_count = phones.Count,
                bert_shape = new[] { bert.Length, bert.Length > 0 ? bert[0].Length : 0 },
                norm_text = normText,
                // Save a few bert values for comparison
                bert_sample = bert.Length > 0 && bert[0].Length > 0
                    ? bert[0].Take(Math.Min(5, bert[0].Length)).ToArray()
                    : Array.Empty<float>()
            }
        };

        // Create test_results directory if it doesn't exist
        var resultsDir = Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..", "..", "..", "test_results");
        resultsDir = Path.GetFullPath(resultsDir);
        Directory.CreateDirectory(resultsDir);

        var filename = Path.Combine(resultsDir, $"csharp_{testName}.json");
        var json = JsonSerializer.Serialize(result, new JsonSerializerOptions
        {
            WriteIndented = true,
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
        });

        File.WriteAllText(filename, json);
        Console.WriteLine($"Saved to: {filename}");
    }
}
