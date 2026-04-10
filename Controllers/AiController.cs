using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using OpenAI.Chat;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

[ApiController]
[Route("api/[controller]")]
public class AiController : ControllerBase
{
    private readonly ChatClient _chatClient;

    public AiController(IConfiguration config)
    {
        var key = config["OpenAI:ApiKey"]!;
        _chatClient = new ChatClient("gpt-4o", key);
    }

    public record ElementPayload(
        string Name,
        string Category,
        string? Family,
        string? Type,
        Dictionary<string, string> Properties
    );

    [HttpPost("explain")]
    public async Task Explain([FromBody] ElementPayload element)
    {
        Response.ContentType = "text/plain; charset=utf-8";
        Response.Headers["Cache-Control"] = "no-cache";

        try
        {
            var propLines = element.Properties
                .Select(kv => $"  - {kv.Key}: {kv.Value}")
                .Take(40); // cap to avoid token explosion

            var hasFireRating = element.Properties.TryGetValue("Fire Rating", out var fireRating)
                    && !string.IsNullOrWhiteSpace(fireRating)
                    && fireRating != "0";

            var hasFlow = element.Properties.TryGetValue("Flow", out var flowValue)
                          && !string.IsNullOrWhiteSpace(flowValue)
                          && flowValue != "0";

            var prompt = $"""
                You are a BIM expert assistant analyzing a Revit model element.
    
                Respond in this exact structure:

                📋 ELEMENT SUMMARY
                One sentence describing what this element is and its function in the building.

                🔧 KEY PROPERTIES
                List the most relevant technical properties from the data below (flow rate, pressure, system name, 
                level, size, rating, capacity, voltage, wattage — whatever is applicable to this category).
                Format each as:  • Property Name: value [unit if known]
                Skip properties that are empty, zero, or irrelevant to this element type.

                ⚠️ MISSING / ISSUES
                List any critical parameters that are missing or empty (system assignment, level, etc.)
                List any likely code or standard violations visible from the data.
                {(!hasFireRating ? "⛔ Fire Rating is not set on this element — this is a critical omission. Flag it explicitly." : "")}
                {(!hasFlow ? "⛔ Flow is null or zero — this element has no flow assigned. Flag it explicitly as a potential design or data issue." : "")}
                If nothing is missing or wrong, write "None identified."

                ✅ RECOMMENDATION
                One concrete, actionable next step for the BIM author or engineer.

                ---
                Element data:
                Name: {element.Name}
                Category: {element.Category}
                Family: {element.Family ?? "unknown"}
                Type: {element.Type ?? "unknown"}
                Properties:
                {string.Join("\n", propLines)}

                Rules:
                - Be specific and technical (MEP/structural/architectural language as appropriate)
                - Only list properties that have meaningful non-zero values in KEY PROPERTIES
                - Keep total response under 200 words
                - Do not invent values not present in the data
                """;

            var messages = new List<ChatMessage>
            {
                new SystemChatMessage("You are a BIM QA assistant. Be direct, technical, and actionable."),
                new UserChatMessage(prompt)
            };

            await foreach (var chunk in _chatClient.CompleteChatStreamingAsync(messages))
            {
                foreach (var part in chunk.ContentUpdate)
                {
                    await Response.WriteAsync(part.Text);
                    await Response.Body.FlushAsync();
                }
            }

        }
        catch (System.Exception ex)
        {
            await Response.WriteAsync($"\n\n❌ AI Error: {ex.Message}");
            await Response.Body.FlushAsync();
        }
    }

    public record FollowUpPayload(
    string Name,
    string Category,
    string? Family,
    string? Type,
    Dictionary<string, string> Properties,
    string FollowUp,
    List<FollowUpMessage>? History
);

    public record FollowUpMessage(string Role, string Content);

    [HttpPost("followup")]
    public async Task FollowUp([FromBody] FollowUpPayload payload)
    {
        Response.ContentType = "text/plain; charset=utf-8";
        Response.Headers["Cache-Control"] = "no-cache";

        var messages = new List<ChatMessage>
    {
        new SystemChatMessage(
            "You are a BIM QA assistant. The user is asking about a specific Revit element. " +
            "Be direct, technical, and keep answers under 120 words.")
    };

        // Re-inject element summary so the model has context
        messages.Add(new UserChatMessage(
            $"I'm looking at element '{payload.Name}' (Category: {payload.Category}, " +
            $"Family: {payload.Family}, Type: {payload.Type})."));

        // Replay conversation history
        if (payload.History != null)
            foreach (var h in payload.History)
                messages.Add(h.Role == "assistant"
                    ? new AssistantChatMessage(h.Content)
                    : new UserChatMessage(h.Content));

        messages.Add(new UserChatMessage(payload.FollowUp));

        await foreach (var chunk in _chatClient.CompleteChatStreamingAsync(messages))
            foreach (var part in chunk.ContentUpdate)
            {
                await Response.WriteAsync(part.Text);
                await Response.Body.FlushAsync();
            }
    }
}