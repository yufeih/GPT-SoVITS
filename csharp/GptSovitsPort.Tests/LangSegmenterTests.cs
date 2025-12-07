using System.Text.Json;
using Xunit;
using Xunit.Abstractions;

namespace GptSovitsPort.Tests;

public class LangSegmenterTests
{
    private readonly ITestOutputHelper _output;

    public LangSegmenterTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private class TestCase
    {
        public string Name { get; set; } = "";
        public string Text { get; set; } = "";
        public string DefaultLang { get; set; } = "";
        public List<LangSegment>? Expected { get; set; }
    }

    private List<TestCase> GetTestCases()
    {
        return new List<TestCase>
        {
            // Basic language detection
            new() {
                Name = "pure_english",
                Text = "Hello world this is a test",
                DefaultLang = "",
                Expected = new() { new() { Lang = "en", Text = "Hello world this is a test" } }
            },
            new() {
                Name = "pure_chinese",
                Text = "这是一个测试",
                DefaultLang = "",
                Expected = new() { new() { Lang = "zh", Text = "这是一个测试" } }
            },
            new() {
                Name = "pure_japanese",
                Text = "これはテストです",
                DefaultLang = "",
                Expected = new() { new() { Lang = "ja", Text = "これはテストです" } }
            },
            new() {
                Name = "pure_korean",
                Text = "이것은 테스트입니다",
                DefaultLang = "",
                Expected = new() { new() { Lang = "ko", Text = "이것은 테스트입니다" } }
            },

            // Mixed language
            new() {
                Name = "mixed_ja_en",
                Text = "MyGO?,你也喜欢まいご吗？",
                DefaultLang = "",
            },
            new() {
                Name = "mixed_ja_sentence",
                Text = "ねえ、知ってる？最近、僕は天文学を勉強してるんだ。君の瞳が星空みたいにキラキラしてるからさ。",
                DefaultLang = "",
            },

            // With default_lang
            new() {
                Name = "mixed_with_default_zh",
                Text = "当时ThinkPad T60刚刚发布，一同推出的还有一款名为Advanced Dock的扩展坞配件。",
                DefaultLang = "zh",
            },
            new() {
                Name = "mixed_without_default",
                Text = "当时ThinkPad T60刚刚发布，一同推出的还有一款名为Advanced Dock的扩展坞配件。",
                DefaultLang = "",
            },

            // Numbers and digits
            new() {
                Name = "text_with_numbers",
                Text = "I have 123 apples and 456 oranges",
                DefaultLang = "",
            },
            new() {
                Name = "chinese_with_numbers",
                Text = "我有123个苹果和456个橙子",
                DefaultLang = "",
            },

            // Short English detection
            new() {
                Name = "short_english_words",
                Text = "这是OK的做法",
                DefaultLang = "",
            },

            // Japanese in non-Japanese text
            new() {
                Name = "japanese_in_chinese",
                Text = "中文まいご中文",
                DefaultLang = "",
            },

            // Korean in non-Korean text
            new() {
                Name = "korean_in_chinese",
                Text = "中文한글中文",
                DefaultLang = "",
            },

            // CJK detection
            new() {
                Name = "mixed_cjk",
                Text = "中文日本語한글",
                DefaultLang = "",
            },

            // Digit edge cases
            new() {
                Name = "digit_at_start",
                Text = "123 is a number",
                DefaultLang = "",
            },
            new() {
                Name = "digit_at_end",
                Text = "The number is 123",
                DefaultLang = "",
            },
            new() {
                Name = "digit_in_middle",
                Text = "I have 123 items here",
                DefaultLang = "",
            },

            // Multiple punctuation scenarios for digit lang detection
            new() {
                Name = "digit_with_punctuation_before",
                Text = "Hello, 123 world",
                DefaultLang = "",
            },
            new() {
                Name = "digit_with_punctuation_after",
                Text = "Hello 123, world",
                DefaultLang = "",
            },
        };
    }

    [Fact]
    public void TestAllCases()
    {
        var testCases = GetTestCases();
        var results = new List<object>();

        foreach (var testCase in testCases)
        {
            var result = LangSegmenter.GetTexts(testCase.Text, testCase.DefaultLang);

            var testResult = new
            {
                name = testCase.Name,
                input = new
                {
                    text = testCase.Text,
                    default_lang = testCase.DefaultLang
                },
                output = result.Select(s => new { lang = s.Lang, text = s.Text }).ToList()
            };

            results.Add(testResult);

            _output.WriteLine($"\n{testCase.Name}:");
            _output.WriteLine($"  Input: text='{testCase.Text}', default_lang='{testCase.DefaultLang}'");
            _output.WriteLine($"  Output: {JsonSerializer.Serialize(result.Select(s => new { lang = s.Lang, text = s.Text }))}");

            if (testCase.Expected != null)
            {
                Assert.Equal(testCase.Expected.Count, result.Count);
                for (int i = 0; i < testCase.Expected.Count; i++)
                {
                    Assert.Equal(testCase.Expected[i].Lang, result[i].Lang);
                    Assert.Equal(testCase.Expected[i].Text, result[i].Text);
                }
                _output.WriteLine($"  Match: True");
            }
        }

        // Save results to JSON
        var outputDir = Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..", "..", "..", "tests", "test_results");
        Directory.CreateDirectory(outputDir);

        var outputFile = Path.Combine(outputDir, "lang_segmenter_results_csharp.json");
        var json = JsonSerializer.Serialize(results, new JsonSerializerOptions
        {
            WriteIndented = true,
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
        });
        File.WriteAllText(outputFile, json);

        _output.WriteLine($"\n\nResults saved to: {outputFile}");
        _output.WriteLine($"Total test cases: {testCases.Count}");
    }
}
