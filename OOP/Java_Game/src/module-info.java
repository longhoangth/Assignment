module application {
    requires javafx.controls;
    requires javafx.fxml;
    exports application;
    requires transitive javafx.graphics;
}