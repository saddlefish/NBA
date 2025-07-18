# Load libraries
library(shiny)
library(ggplot2)
library(dplyr)
library(plotly)

# Load data
shot_data <- read.csv("enhanced_shot_data.csv")

# Define UI
ui <- fluidPage(
  titlePanel("NBA Shot Selection Optimization Dashboard"),
  sidebarLayout(
    sidebarPanel(
      selectInput("player", "Select Player:", choices = unique(shot_data$PLAYER_NAME)),
      sliderInput("prob_threshold", "Shot Probability Threshold:", min = 0, max = 1, value = 0.5, step = 0.05),
      h4("Instructions"),
      p("Select a player and probability threshold to view shot efficiency and recommendations.")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Shot Location Heatmap", plotlyOutput("heatmap")),
        tabPanel("Shot Success by Zone", plotOutput("zone_plot")),
        tabPanel("Expected Points", plotOutput("points_plot"))
      )
    )
  )
)

# Define server
server <- function(input, output, session) {
  # Reactive data filtering
  filtered_data <- reactive({
    shot_data %>%
      filter(PLAYER_NAME == input$player) %>%
      mutate(RECOMMENDED = SHOT_PROB > input$prob_threshold)
  })
  
  # Heatmap of shot locations
  output$heatmap <- renderPlotly({
    data <- filtered_data()
    plot_ly(data, x = ~LOC_X, y = ~LOC_Y, type = 'scatter', mode = 'markers',
            color = ~SHOT_PROB, colors = "RdYlBu",
            size = 5, alpha = 0.6,
            text = ~paste("Shot Type:", SHOT_TYPE, "<br>Prob:", round(SHOT_PROB, 2))) %>%
      layout(title = "Shot Locations with Success Probability",
             xaxis = list(title = "Court X (feet)"),
             yaxis = list(title = "Court Y (feet)"))
  })
  
  # Bar plot of success rate by zone
  output$zone_plot <- renderPlot({
    data <- filtered_data() %>%
      group_by(SHOT_ZONE_BASIC) %>%
      summarise(Success_Rate = mean(SHOT_MADE_FLAG), .groups = 'drop')
    ggplot(data, aes(x = factor(SHOT_ZONE_BASIC), y = Success_Rate)) +
      geom_bar(stat = "identity", fill = "steelblue") +
      theme_minimal() +
      labs(title = "Shot Success Rate by Zone", x = "Shot Zone (Encoded)", y = "Success Rate")
  })
  
  # Bar plot of expected points
  output$points_plot <- renderPlot({
    data <- filtered_data()
    actual_points <- mean(data$SHOT_MADE_FLAG * data$SHOT_TYPE, na.rm = TRUE)
    recommended_points <- mean(data$EXPECTED_POINTS[data$RECOMMENDED], na.rm = TRUE)
    points_data <- data.frame(
      Group = c("Actual Shots", "Recommended Shots"),
      Points = c(actual_points, recommended_points)
    )
    ggplot(points_data, aes(x = Group, y = Points, fill = Group)) +
      geom_bar(stat = "identity") +
      theme_minimal() +
      labs(title = "Expected Points per Shot", x = "", y = "Points")
  })
}

# Run the app
shinyApp(ui = ui, server = server)
