import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:sentinel_ai/main.dart';
import 'package:sentinel_ai/providers/sensor_provider.dart';

void main() {
  testWidgets('Dashboard renders successfully', (WidgetTester tester) async {
    // Build our app and trigger a frame
    await tester.pumpWidget(
      ChangeNotifierProvider(
        create: (_) => SensorProvider(),
        child: const SentinelApp(),
      ),
    );

    // Verify SentinelAI header title is displayed in the AppBar
    expect(find.text('SentinelAI'), findsOneWidget);
    
    // Verify that the interaction canvas instruction text is rendered
    expect(find.text('Perform a tap or swipe in this canvas area'), findsOneWidget);
  });
}
