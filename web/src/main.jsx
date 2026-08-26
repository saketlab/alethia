import React from 'react';
import { createRoot } from 'react-dom/client';
import { Theme } from '@radix-ui/themes';
import '@radix-ui/themes/styles.css';

import App from './ui/App.jsx';
import { useAppearance } from './ui/useAppearance.js';
import './ui/app.css';

function Root() {
  const { preference, appearance, choose } = useAppearance();
  return (
    // appearance follows the OS unless the user has chosen otherwise
    <Theme
      appearance={appearance}
      accentColor="indigo"
      grayColor="slate"
      radius="large"
      scaling="100%"
    >
      <App appearance={{ preference, choose }} />
    </Theme>
  );
}

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
);
